//! Convergence oracle and hardening tests for the collaboration layer.
//! The load-bearing property is CONVERGENCE: any set of replicas that have
//! seen the same events materialize byte-identical documents, regardless of
//! merge order, batch splitting, or gossip topology. Secondary gates: the
//! returned edit streams are valid (in-bounds, length-consistent — they are
//! the caller's anchor-shifting input), the wire format round-trips, and
//! nothing leaks under allocation failure.

const std = @import("std");
const t = std.testing;

const TextDoc = @import("TextDoc.zig");
const geometry = @import("../geometry.zig");
const AnchorSet = @import("../AnchorSet.zig");

const EventId = TextDoc.EventId;
const Range = geometry.Range;
const Edit = geometry.Edit;

fn docText(gpa: std.mem.Allocator, d: *const TextDoc) ![]u8 {
    return d.text().toOwnedSlice(gpa);
}

fn expectDocText(d: *const TextDoc, expected: []const u8) !void {
    const got = try docText(t.allocator, d);
    defer t.allocator.free(got);
    try t.expectEqualStrings(expected, got);
}

/// One-direction sync: bring `to` up to date with `from`. Returns the edit
/// stream `to` observed (caller frees), after validating it.
fn syncOne(gpa: std.mem.Allocator, from: *const TextDoc, to: *TextDoc) ![]Edit {
    const ver = try to.version(gpa);
    defer gpa.free(ver);
    const batch = try from.eventsSince(gpa, ver);
    defer gpa.free(batch);
    const pre_len = to.text().byteLen();
    const edits = try to.merge(gpa, batch);
    // Edit-stream validity: each edit in-bounds at its application point;
    // lengths reconcile exactly with the document's length change.
    var len = pre_len;
    for (edits) |e| {
        try t.expect(e.offset <= len);
        try t.expect(e.offset + e.removed <= len);
        len = len - e.removed + e.inserted;
    }
    try t.expectEqual(to.text().byteLen(), len);
    return edits;
}

fn syncBoth(gpa: std.mem.Allocator, a: *TextDoc, b: *TextDoc) !void {
    gpa.free(try syncOne(gpa, a, b));
    gpa.free(try syncOne(gpa, b, a));
}

fn expectConverged(docs: []const *TextDoc) !void {
    const gpa = t.allocator;
    const first = try docText(gpa, docs[0]);
    defer gpa.free(first);
    for (docs[1..]) |d| {
        const other = try docText(gpa, d);
        defer gpa.free(other);
        try t.expectEqualStrings(first, other);
    }
}

test "linear sync: one author, one reader" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");

    _ = try alice.insert(gpa, 0, "hello world");
    _ = try alice.delete(gpa, .{ .start = 5, .end = 11 });
    _ = try alice.insert(gpa, 5, ", stemma");

    try syncBoth(gpa, &alice, &bob);
    try expectDocText(&bob, "hello, stemma");
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    // Idempotence: replaying the same batch changes nothing.
    const all = try alice.serialize(gpa);
    defer gpa.free(all);
    const edits = try bob.merge(gpa, all);
    defer gpa.free(edits);
    try t.expectEqual(@as(usize, 0), edits.len);
    try expectDocText(&bob, "hello, stemma");
}

test "concurrent edits converge; remote edits shift anchors correctly" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "shared base text");
    try syncBoth(gpa, &alice, &bob);

    // Bob holds a caret on "base" (offset 7) and a mark at the end.
    var bob_anchors: AnchorSet = .empty;
    defer bob_anchors.deinit(gpa);
    const caret = try bob_anchors.add(gpa, .{ .offset = 7, .bias = .left });
    const mark = try bob_anchors.add(gpa, .{ .offset = 16, .bias = .right });

    // Concurrent: alice prepends; bob appends.
    _ = try alice.insert(gpa, 0, "[alice] ");
    const bob_edit = try bob.insert(gpa, 16, "!");
    bob_anchors.shift(bob_edit);

    // Bob merges alice's concurrent prepend; his anchors ride the stream.
    const remote_edits = try syncOne(gpa, &alice, &bob);
    defer gpa.free(remote_edits);
    for (remote_edits) |e| bob_anchors.shift(e);
    gpa.free(try syncOne(gpa, &bob, &alice));

    try expectConverged(&[_]*TextDoc{ &alice, &bob });
    try expectDocText(&bob, "[alice] shared base text!");

    // The caret still points at "base"; the mark rode past the bang.
    const bob_text = try docText(gpa, &bob);
    defer gpa.free(bob_text);
    const caret_off = bob_anchors.get(caret).offset;
    try t.expectEqualStrings("base", bob_text[caret_off..][0..4]);
    try t.expectEqual(bob_text.len, bob_anchors.get(mark).offset);
}

test "concurrent runs do not interleave" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "()");
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.insert(gpa, 1, "aaaa");
    _ = try bob.insert(gpa, 1, "bbbb");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    const got = try docText(gpa, &alice);
    defer gpa.free(got);
    const ok = std.mem.eql(u8, got, "(aaaabbbb)") or std.mem.eql(u8, got, "(bbbbaaaa)");
    try t.expect(ok);
}

test "backward interleaving: concurrent prepend blocks stay contiguous (FugueMax)" {
    // The anomaly that separates FugueMax from plain Yjs: two peers each
    // building a block by repeatedly inserting at position 0. Under Yjs the
    // blocks can interleave; under FugueMax each block stays contiguous.
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "|");
    try syncBoth(gpa, &alice, &bob);

    // Each peer prepends three units, one at a time, all at position 0.
    for ("cba") |c| _ = try alice.insert(gpa, 0, &.{c});
    for ("zyx") |c| _ = try bob.insert(gpa, 0, &.{c});
    // alice: "abc|", bob: "xyz|"
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    const got = try docText(gpa, &alice);
    defer gpa.free(got);
    const ok = std.mem.eql(u8, got, "abcxyz|") or std.mem.eql(u8, got, "xyzabc|");
    if (!ok) std.debug.print("interleaved: {s}\n", .{got});
    try t.expect(ok);
}

test "forward interleaving: concurrent append blocks stay contiguous" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "|");
    try syncBoth(gpa, &alice, &bob);
    for ("abc", 1..) |c, i| _ = try alice.insert(gpa, i, &.{c});
    for ("xyz", 1..) |c, i| _ = try bob.insert(gpa, i, &.{c});
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    const got = try docText(gpa, &alice);
    defer gpa.free(got);
    const ok = std.mem.eql(u8, got, "|abcxyz") or std.mem.eql(u8, got, "|xyzabc");
    if (!ok) std.debug.print("interleaved: {s}\n", .{got});
    try t.expect(ok);
}

test "concurrent delete + insert inside the deleted span" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "keep DELETE keep");
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.delete(gpa, .{ .start = 5, .end = 12 }); // "DELETE "
    _ = try bob.insert(gpa, 8, "X"); // inside alice's doomed span
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    // Bob's X survives (it was not deleted — only alice's chars were).
    const got = try docText(gpa, &alice);
    defer gpa.free(got);
    try t.expect(std.mem.indexOfScalar(u8, got, 'X') != null);
    try t.expect(std.mem.indexOf(u8, got, "DELETE") == null);
}

test "double delete converges to a single deletion" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "abcdef");
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.delete(gpa, .{ .start = 2, .end = 4 }); // both delete "cd"
    _ = try bob.delete(gpa, .{ .start = 2, .end = 4 });
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });
    try expectDocText(&alice, "abef");
}

test "batch splitting is irrelevant: one merge == event-by-event merges" {
    const gpa = t.allocator;
    var author: TextDoc = .empty;
    defer author.deinit(gpa);
    try author.setAgent(gpa, "author");
    _ = try author.insert(gpa, 0, "batch splitting test — 分割 œuvre\n");
    _ = try author.delete(gpa, .{ .start = 0, .end = 6 });
    _ = try author.insert(gpa, 0, "BATCH ");

    const whole = try author.serialize(gpa);
    defer gpa.free(whole);

    var one_shot = try TextDoc.open(gpa, whole);
    defer one_shot.deinit(gpa);

    // Same events, merged one unit at a time via incremental eventsSince.
    var stepwise: TextDoc = .empty;
    defer stepwise.deinit(gpa);
    while (true) {
        const ver = try stepwise.version(gpa);
        defer gpa.free(ver);
        const missing = try author.eventsSince(gpa, ver);
        defer gpa.free(missing);
        const edits = try stepwise.merge(gpa, missing);
        const done = edits.len == 0;
        gpa.free(edits);
        if (done) break;
    }
    try expectConverged(&[_]*TextDoc{ &author, &one_shot, &stepwise });
}

test "serialize/open roundtrip preserves version and future syncability" {
    const gpa = t.allocator;
    var orig: TextDoc = .empty;
    defer orig.deinit(gpa);
    try orig.setAgent(gpa, "orig");
    _ = try orig.insert(gpa, 0, "persisted 内容 with unicode 𝄞");
    _ = try orig.delete(gpa, .{ .start = 0, .end = 4 });

    const bytes = try orig.serialize(gpa);
    defer gpa.free(bytes);
    var reopened = try TextDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try expectConverged(&[_]*TextDoc{ &orig, &reopened });

    const va = try orig.version(gpa);
    defer gpa.free(va);
    const vb = try reopened.version(gpa);
    defer gpa.free(vb);
    try t.expectEqualSlices(u8, va, vb);

    // The reopened doc keeps collaborating.
    try reopened.setAgent(gpa, "reopened");
    _ = try reopened.insert(gpa, 0, ">> ");
    try syncBoth(gpa, &reopened, &orig);
    try expectConverged(&[_]*TextDoc{ &orig, &reopened });
}

test "corrupt and causally-incomplete input is rejected without damage" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    _ = try alice.insert(gpa, 0, "stable");
    const before = try docText(gpa, &alice);
    defer gpa.free(before);

    try t.expectError(error.Corrupt, alice.merge(gpa, "not a stemma batch"));

    // A valid batch minus its first event: dangling causal dependency.
    var author: TextDoc = .empty;
    defer author.deinit(gpa);
    try author.setAgent(gpa, "author");
    _ = try author.insert(gpa, 0, "xy");
    const full = try author.serialize(gpa);
    defer gpa.free(full);
    var partial: TextDoc = .empty;
    defer partial.deinit(gpa);
    // Events "since seq 0" delivered to a doc that has nothing: the batch's
    // first event depends on an event the receiver lacks. Version token
    // hand-built (opaque format: magic, count, name_len, name, seq).
    const claim_seq0 = "stv\x01" ++ [_]u8{ 1, 6 } ++ "author" ++ [_]u8{0};
    const gap_batch = try author.eventsSince(gpa, claim_seq0);
    defer gpa.free(gap_batch);
    try t.expectError(error.MissingDependency, partial.merge(gpa, gap_batch));

    try expectDocText(&alice, before);
}

test "malicious batch: out-of-range position is rejected atomically" {
    const gpa = t.allocator;
    var victim: TextDoc = .empty;
    defer victim.deinit(gpa);
    try victim.setAgent(gpa, "victim");
    _ = try victim.insert(gpa, 0, "stable");
    const before_events = victim.history.eventCount();

    // Hand-crafted batch: agent "evil", one event, seq 0, no parents,
    // insert at position 999 (LEB128: 0xE7 0x07) — far out of range.
    const evil = "stg\x01" ++ [_]u8{ 1, 4 } ++ "evil" ++
        [_]u8{ 1, 0, 0, 0, 0 } ++ [_]u8{ 0xE7, 0x07 } ++ [_]u8{'x'};
    try t.expectError(error.Corrupt, victim.merge(gpa, evil));

    // Fully atomic: no events retained, document intact, still functional.
    try t.expectEqual(before_events, victim.history.eventCount());
    try expectDocText(&victim, "stable");
    _ = try victim.insert(gpa, 6, "!");
    try expectDocText(&victim, "stable!");
}

test "malicious batch: out-of-range delete is rejected atomically" {
    const gpa = t.allocator;
    var victim: TextDoc = .empty;
    defer victim.deinit(gpa);
    try victim.setAgent(gpa, "victim");
    _ = try victim.insert(gpa, 0, "ab");
    // Delete at position 7 in a 2-scalar doc.
    const evil = "stg\x01" ++ [_]u8{ 1, 4 } ++ "evil" ++
        [_]u8{ 1, 0, 0, 0, 1, 7 };
    try t.expectError(error.Corrupt, victim.merge(gpa, evil));
    try expectDocText(&victim, "ab");
}

fn fuzzWire(_: void, smith: *std.testing.Smith) !void {
    const gpa = t.allocator;
    var author: TextDoc = .empty;
    defer author.deinit(gpa);
    try author.setAgent(gpa, "author");
    _ = try author.insert(gpa, 0, "wire fuzz корпус 𝄞\n");
    _ = try author.delete(gpa, .{ .start = 2, .end = 6 });
    const valid = try author.serialize(gpa);
    defer gpa.free(valid);

    // Mutate a few bytes of a valid batch; merge must never crash or leak —
    // any outcome in {success, Corrupt, MissingDependency} is acceptable.
    const mutated = try gpa.dupe(u8, valid);
    defer gpa.free(mutated);
    for (0..1 + smith.indexWithHash(4, 0x0f11)) |_| {
        const at = smith.indexWithHash(mutated.len, 0x0a7e);
        mutated[at] = smith.valueWithHash(u8, 0xb17e);
    }
    var victim: TextDoc = .empty;
    defer victim.deinit(gpa);
    if (victim.merge(gpa, mutated)) |edits| {
        gpa.free(edits);
    } else |err| switch (err) {
        error.Corrupt, error.MissingDependency => {
            // Rejected batches must leave the graph untouched.
            try t.expectEqual(@as(usize, 0), victim.history.eventCount());
        },
        else => |e| return e,
    }
}

test "fuzz: mutated wire bytes never crash, rejects are atomic" {
    try std.testing.fuzz({}, fuzzWire, .{});
}

test "three peers, seeded random gossip, full convergence" {
    const gpa = t.allocator;
    const names = [_][]const u8{ "alice", "bob", "carol" };
    var docs: [3]TextDoc = .{ .empty, .empty, .empty };
    defer for (&docs) |*d| d.deinit(gpa);
    for (&docs, names) |*d, n| try d.setAgent(gpa, n);

    const pool = [_][]const u8{ "a", "xy", "héllo", "𝄞", "\n", "word " };

    for ([_]u64{ 1, 42, 0xbeef }) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        for (0..24) |_| {
            // A random peer performs a few random local edits.
            const d = &docs[random.uintLessThan(usize, docs.len)];
            for (0..1 + random.uintLessThan(usize, 3)) |_| {
                const len = d.text().byteLen();
                const do_insert = len < 8 or random.boolean();
                if (do_insert) {
                    const scalars = d.text().scalarLen();
                    const pos = d.text().scalarToOffset(random.uintLessThan(usize, scalars + 1));
                    _ = try d.insert(gpa, pos, pool[random.uintLessThan(usize, pool.len)]);
                } else {
                    const scalars = d.text().scalarLen();
                    const s0 = random.uintLessThan(usize, scalars);
                    const s1 = @min(scalars, s0 + 1 + random.uintLessThan(usize, 4));
                    _ = try d.delete(gpa, .{
                        .start = d.text().scalarToOffset(s0),
                        .end = d.text().scalarToOffset(s1),
                    });
                }
            }
            // Random one-directional gossip.
            const i = random.uintLessThan(usize, docs.len);
            var j = random.uintLessThan(usize, docs.len);
            if (i == j) j = (j + 1) % docs.len;
            gpa.free(try syncOne(gpa, &docs[i], &docs[j]));
        }

        // Full mesh sync until fixpoint, then everyone agrees.
        for (0..2) |_| {
            for (0..docs.len) |i| {
                for (0..docs.len) |j| {
                    if (i != j) gpa.free(try syncOne(gpa, &docs[i], &docs[j]));
                }
            }
        }
        try expectConverged(&[_]*TextDoc{ &docs[0], &docs[1], &docs[2] });
    }
}

test "compareVersions: equal, ancestor, descendant, concurrent" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "base");
    const v1 = try alice.version(gpa);
    defer gpa.free(v1);
    try syncBoth(gpa, &alice, &bob);

    try t.expectEqual(.equal, try alice.compareVersions(gpa, v1, v1));

    _ = try alice.insert(gpa, 0, "more");
    const v2 = try alice.version(gpa);
    defer gpa.free(v2);
    try t.expectEqual(.ancestor, try alice.compareVersions(gpa, v1, v2));
    try t.expectEqual(.descendant, try alice.compareVersions(gpa, v2, v1));

    // Bob edits concurrently; alice learns of it, then compares.
    _ = try bob.insert(gpa, 4, "!");
    const vb = try bob.version(gpa);
    defer gpa.free(vb);
    gpa.free(try syncOne(gpa, &bob, &alice));
    try t.expectEqual(.concurrent, try alice.compareVersions(gpa, v2, vb));

    // Tokens naming events we don't have are honest errors.
    const unknown = "stv\x01" ++ [_]u8{ 1, 5 } ++ "ghost" ++ [_]u8{3};
    try t.expectError(error.MissingDependency, alice.compareVersions(gpa, v1, unknown));
}

test "version tokens canonically order concurrent heads across replicas" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    // Each replica creates one concurrent event first.  Merging in opposite
    // directions gives the two graphs opposite local Lv orders while leaving
    // them with the same portable frontier {alice#0, bob#0}.
    _ = try alice.insert(gpa, 0, "a");
    _ = try bob.insert(gpa, 0, "b");
    const alice_events = try alice.serialize(gpa);
    defer gpa.free(alice_events);
    const bob_events = try bob.serialize(gpa);
    defer gpa.free(bob_events);

    const alice_edits = try alice.merge(gpa, bob_events);
    defer gpa.free(alice_edits);
    const bob_edits = try bob.merge(gpa, alice_events);
    defer gpa.free(bob_edits);

    const alice_version = try alice.version(gpa);
    defer gpa.free(alice_version);
    const bob_version = try bob.version(gpa);
    defer gpa.free(bob_version);
    try t.expectEqualSlices(u8, alice_version, bob_version);
}

test "materializeAt: time travel to any known version" {
    const gpa = t.allocator;
    var d: TextDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");

    // Empty version → empty document.
    const v_empty = try d.version(gpa);
    defer gpa.free(v_empty);
    var at_empty = try d.materializeAt(gpa, v_empty);
    defer at_empty.deinit(gpa);
    try t.expect(at_empty.isEmpty());

    _ = try d.insert(gpa, 0, "first draft 𝄞");
    const v1 = try d.version(gpa);
    defer gpa.free(v1);
    const text_v1 = try docText(gpa, &d);
    defer gpa.free(text_v1);

    _ = try d.delete(gpa, .{ .start = 0, .end = 6 });
    _ = try d.insert(gpa, 0, "FINAL");
    const v2 = try d.version(gpa);
    defer gpa.free(v2);

    var at_v1 = try d.materializeAt(gpa, v1);
    defer at_v1.deinit(gpa);
    const got_v1 = try at_v1.toOwnedSlice(gpa);
    defer gpa.free(got_v1);
    try t.expectEqualStrings(text_v1, got_v1);

    var at_v2 = try d.materializeAt(gpa, v2);
    defer at_v2.deinit(gpa);
    try t.expect(at_v2.eql(d.rope));

    // A version we've never seen is a missing dependency.
    const unknown = "stv\x01" ++ [_]u8{ 1, 5 } ++ "ghost" ++ [_]u8{9};
    try t.expectError(error.MissingDependency, d.materializeAt(gpa, unknown));
}

test "identity anchors: survive concurrent merges across replicas" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "hello World");
    try syncBoth(gpa, &alice, &bob);

    // Alice anchors her cursor before the 'W'.
    const a = try alice.anchorAt(gpa, 6, .before);
    defer gpa.free(a.agent);

    // Divergent concurrent edits on both sides.
    _ = try alice.insert(gpa, 0, ">>> ");
    _ = try bob.insert(gpa, 5, ", cruel");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    // Both replicas resolve the SAME anchor (portable: name+seq) and land
    // on the 'W', wherever it now lives in each doc.
    for ([_]*TextDoc{ &alice, &bob }) |d| {
        var off: [1]usize = undefined;
        try d.resolveAnchors(gpa, &.{a}, &off);
        const txt = try docText(gpa, d);
        defer gpa.free(txt);
        try t.expectEqual(@as(u8, 'W'), txt[off[0]]);
    }

    // Deleting the target collapses the anchor to the deletion point.
    var off_before: [1]usize = undefined;
    try bob.resolveAnchors(gpa, &.{a}, &off_before);
    _ = try bob.delete(gpa, .{ .start = off_before[0], .end = off_before[0] + 5 });
    var off_after: [1]usize = undefined;
    try bob.resolveAnchors(gpa, &.{a}, &off_after);
    try t.expectEqual(off_before[0], off_after[0]);

    // Boundary anchors.
    const start = try bob.anchorAt(gpa, 0, .after);
    const end = try bob.anchorAt(gpa, bob.text().byteLen(), .before);
    var offs: [2]usize = undefined;
    try bob.resolveAnchors(gpa, &.{ start, end }, &offs);
    try t.expectEqual(@as(usize, 0), offs[0]);
    try t.expectEqual(bob.text().byteLen(), offs[1]);
}

test "compact: history collapses, text unchanged, collaboration continues" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    // Build shared history from both agents, then a linearization point.
    _ = try alice.insert(gpa, 0, "collaborative 内容 here");
    try syncBoth(gpa, &alice, &bob);
    _ = try bob.insert(gpa, 0, "[bob] ");
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.insert(gpa, alice.text().byteLen(), " (end)"); // single head
    try syncBoth(gpa, &alice, &bob);

    const stable = try alice.version(gpa);
    defer gpa.free(stable);
    const text_before = try docText(gpa, &alice);
    defer gpa.free(text_before);
    const events_before = alice.history.eventCount();

    // Both peers compact at the same stable point.
    try alice.compact(gpa, stable);
    try bob.compact(gpa, stable);
    try t.expectEqual(@as(usize, 0), alice.history.eventCount());
    try t.expect(events_before > 0);
    try expectDocText(&alice, text_before);

    // Editing and syncing continue across the shared base.
    _ = try alice.insert(gpa, 0, "A");
    _ = try bob.insert(gpa, bob.text().byteLen(), "B");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    // Serialize/open round-trips the compacted form (v2 with base), and the
    // reopened doc keeps collaborating.
    const bytes = try alice.serialize(gpa);
    defer gpa.free(bytes);
    var reopened = try TextDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try expectConverged(&[_]*TextDoc{ &alice, &reopened });
    try reopened.setAgent(gpa, "carol");
    _ = try reopened.insert(gpa, 0, "C");
    try syncBoth(gpa, &reopened, &alice);
    try expectConverged(&[_]*TextDoc{ &alice, &reopened });
}

test "compact: mid-history point keeps later events working" {
    const gpa = t.allocator;
    var d: TextDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");
    _ = try d.insert(gpa, 0, "early work…");
    const mid = try d.version(gpa);
    defer gpa.free(mid);
    _ = try d.insert(gpa, d.text().byteLen(), " later work");
    _ = try d.delete(gpa, .{ .start = 0, .end = 6 });
    const text_now = try docText(gpa, &d);
    defer gpa.free(text_now);

    try d.compact(gpa, mid);
    try t.expect(d.history.eventCount() > 0); // later events retained
    try expectDocText(&d, text_now);

    // Current version still materializes (with base placeholders).
    const now = try d.version(gpa);
    defer gpa.free(now);
    var at_now = try d.materializeAt(gpa, now);
    defer at_now.deinit(gpa);
    try t.expect(at_now.eql(d.rope));
}

test "compact: rejects multi-head and concurrent-remainder versions" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    // Bob edits against an OLD version of alice's history, so his event is
    // concurrent to alice's later (would-be stable) head.
    _ = try alice.insert(gpa, 0, "a");
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.insert(gpa, 1, "b"); // the later head
    const stale_head = try alice.version(gpa);
    defer gpa.free(stale_head);
    _ = try bob.insert(gpa, 1, "x"); // parents: only "a" — concurrent to "b"
    try syncBoth(gpa, &alice, &bob);

    // Two concurrent heads: not a linearization point.
    const two_heads = try alice.version(gpa);
    defer gpa.free(two_heads);
    try t.expectError(error.NotCompactable, alice.compact(gpa, two_heads));

    // Single head, but bob's retained event anchors to base *interior*
    // (its parent is "a", inside the would-be base, not the head "b").
    try t.expectError(error.NotCompactable, alice.compact(gpa, stale_head));

    // Doc still fully functional after rejections.
    _ = try alice.insert(gpa, 0, "!");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });
}

test "compact: uncompacted peer cannot merge a based batch; anchors into base are Compacted" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "history");
    try syncBoth(gpa, &alice, &bob);
    const stable = try alice.version(gpa);
    defer gpa.free(stable);
    try alice.compact(gpa, stable);
    _ = try alice.insert(gpa, 0, "x");

    // Bob (uncompacted, non-empty) receives alice's compacted batch: reject.
    const vb = try bob.version(gpa);
    defer gpa.free(vb);
    const batch = try alice.eventsSince(gpa, vb);
    defer gpa.free(batch);
    try t.expectError(error.MissingDependency, bob.merge(gpa, batch));

    // After compacting to the same point, sync works again.
    try bob.compact(gpa, stable);
    gpa.free(try syncOne(gpa, &alice, &bob));
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    // Identity anchors cannot target compacted characters…
    try t.expectError(error.Compacted, alice.anchorAt(gpa, 3, .before));
    // …but post-base characters anchor fine.
    const a = try alice.anchorAt(gpa, 0, .before);
    defer gpa.free(a.agent);
    var off: [1]usize = undefined;
    try alice.resolveAnchors(gpa, &.{a}, &off);
    try t.expectEqual(@as(usize, 0), off[0]);
}

fn compactOomScript(gpa: std.mem.Allocator) !void {
    var d: TextDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "oom");
    _ = try d.insert(gpa, 0, "compact under pressure");
    const stable = try d.version(gpa);
    defer gpa.free(stable);
    _ = try d.insert(gpa, 0, "x");
    try d.compact(gpa, stable);
    // Post-compaction the doc still round-trips.
    const bytes = try d.serialize(gpa);
    defer gpa.free(bytes);
    var re = try TextDoc.open(gpa, bytes);
    defer re.deinit(gpa);
}

test "OOM: compaction paths are leak-free under every allocation failure" {
    try std.testing.checkAllAllocationFailures(t.allocator, compactOomScript, .{});
}

fn oomScript(gpa: std.mem.Allocator) !void {
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "base");
    const b1 = try alice.serialize(gpa);
    defer gpa.free(b1);
    gpa.free(try bob.merge(gpa, b1));

    _ = try alice.insert(gpa, 4, "!x");
    _ = try bob.insert(gpa, 0, "y");
    const vb = try bob.version(gpa);
    defer gpa.free(vb);
    const b2 = try alice.eventsSince(gpa, vb);
    defer gpa.free(b2);
    gpa.free(try bob.merge(gpa, b2));
}

test "OOM: collaboration paths are leak-free under every allocation failure" {
    // Documented v1 semantics: the doc may be unusable after mid-merge OOM
    // (rebuild from serialize) — but nothing may leak, ever.
    try std.testing.checkAllAllocationFailures(t.allocator, oomScript, .{});
}

fn fuzzGossip(_: void, smith: *std.testing.Smith) !void {
    const gpa = t.allocator;
    var docs: [2]TextDoc = .{ .empty, .empty };
    defer for (&docs) |*d| d.deinit(gpa);
    try docs[0].setAgent(gpa, "a");
    try docs[1].setAgent(gpa, "b");
    const pool = [_][]const u8{ "z", "qr", "é", "\n" };

    var ops: usize = 0;
    while (ops < 128 and !smith.eosWithHash(0x6055)) : (ops += 1) {
        const d = &docs[smith.indexWithHash(2, 0x0d0c)];
        switch (smith.valueRangeLessThanWithHash(u8, 0, 3, 0x0003)) {
            0 => {
                const scalars = d.text().scalarLen();
                const pos = d.text().scalarToOffset(smith.indexWithHash(scalars + 1, 0x1105));
                _ = try d.insert(gpa, pos, pool[smith.indexWithHash(pool.len, 0x7001)]);
            },
            1 => {
                const scalars = d.text().scalarLen();
                if (scalars == 0) continue;
                const s0 = smith.indexWithHash(scalars, 0xde1);
                _ = try d.delete(gpa, .{
                    .start = d.text().scalarToOffset(s0),
                    .end = d.text().scalarToOffset(s0 + 1),
                });
            },
            2 => {
                const i = smith.indexWithHash(2, 0x5150);
                gpa.free(try syncOne(gpa, &docs[i], &docs[1 - i]));
            },
            else => unreachable,
        }
    }
    try syncBoth(gpa, &docs[0], &docs[1]);
    try expectConverged(&[_]*TextDoc{ &docs[0], &docs[1] });
}

test "fuzz: smith-driven two-peer gossip always converges" {
    try std.testing.fuzz({}, fuzzGossip, .{});
}

// ── Run-RLE wire (v3) ───────────────────────────────────────────────────

test "rle: typing burst is one frame — batch shrinks vs unit encoding" {
    const gpa = t.allocator;
    var author: TextDoc = .empty;
    defer author.deinit(gpa);
    try author.setAgent(gpa, "author");
    _ = try author.insert(gpa, 0, "x" ** 200);
    _ = try author.delete(gpa, .{ .start = 50, .end = 150 });

    const empty_ver = blk: {
        var fresh: TextDoc = .empty;
        defer fresh.deinit(gpa);
        break :blk try fresh.version(gpa);
    };
    defer gpa.free(empty_ver);
    const rle = try author.eventsSinceFormat(gpa, empty_ver, .rle);
    defer gpa.free(rle);
    const unit = try author.eventsSinceFormat(gpa, empty_ver, .unit);
    defer gpa.free(unit);
    // 300 events: two run frames vs 300 unit events. The precise sizes
    // move with the format; the order-of-magnitude win must not.
    try t.expect(rle.len * 4 < unit.len);

    // Both decode to the same document.
    var via_rle = try TextDoc.open(gpa, rle);
    defer via_rle.deinit(gpa);
    var via_unit = try TextDoc.open(gpa, unit);
    defer via_unit.deinit(gpa);
    try expectConverged(&[_]*TextDoc{ &via_rle, &via_unit, &author });
    // Same graph, unit for unit: identical versions and future syncs.
    const va = try via_rle.version(gpa);
    defer gpa.free(va);
    const vb = try via_unit.version(gpa);
    defer gpa.free(vb);
    try t.expectEqualSlices(u8, va, vb);
}

test "rle: mixed runs, singletons, and concurrency converge" {
    const gpa = t.allocator;
    var alice: TextDoc = .empty;
    defer alice.deinit(gpa);
    var bob: TextDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.insert(gpa, 0, "shared base ");
    try syncBoth(gpa, &alice, &bob);
    // Concurrent bursts (runs on both sides) + point edits (singletons).
    _ = try alice.insert(gpa, 12, "alice typed this");
    _ = try alice.delete(gpa, .{ .start = 0, .end = 3 });
    _ = try bob.insert(gpa, 0, "bob's turn: ");
    _ = try bob.insert(gpa, 3, "!");
    _ = try bob.delete(gpa, .{ .start = 1, .end = 2 });
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });

    // Multi-codepoint content in runs survives.
    _ = try alice.insert(gpa, 0, "日本語テキスト");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*TextDoc{ &alice, &bob });
}

test "rle: compacted batch carries base and runs together" {
    const gpa = t.allocator;
    var d: TextDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");
    _ = try d.insert(gpa, 0, "stable prefix");
    const stable = try d.version(gpa);
    defer gpa.free(stable);
    _ = try d.insert(gpa, d.text().byteLen(), " and a long typed suffix after compaction");
    try d.compact(gpa, stable);

    const bytes = try d.serialize(gpa);
    defer gpa.free(bytes);
    try t.expect(std.mem.startsWith(u8, bytes, "stg\x03"));
    var reopened = try TextDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try expectConverged(&[_]*TextDoc{ &d, &reopened });
    // The reopened doc keeps syncing with the original.
    try reopened.setAgent(gpa, "peer");
    _ = try reopened.insert(gpa, 0, "hi ");
    try syncBoth(gpa, &reopened, &d);
    try expectConverged(&[_]*TextDoc{ &d, &reopened });
}

test "rle: hostile run frames are rejected, document untouched" {
    const gpa = t.allocator;
    var author: TextDoc = .empty;
    defer author.deinit(gpa);
    try author.setAgent(gpa, "author");
    _ = try author.insert(gpa, 0, "abcdefgh");

    var victim: TextDoc = .empty;
    defer victim.deinit(gpa);
    const valid = try author.serialize(gpa);
    defer gpa.free(valid);

    // Truncations anywhere must reject cleanly.
    var cut: usize = 4;
    while (cut < valid.len) : (cut += 3) {
        try t.expectError(error.Corrupt, victim.merge(gpa, valid[0..cut]));
    }
    try t.expectEqual(@as(usize, 0), victim.history.eventCount());

    // A delete-run frame claiming an absurd count: tiny input must not
    // expand into memory. count > max_batch_units → Corrupt.
    var evil: std.ArrayList(u8) = .empty;
    defer evil.deinit(gpa);
    try evil.appendSlice(gpa, "stg\x03");
    try evil.append(gpa, 0); // no base
    try evil.appendSlice(gpa, &.{ 1, 1, 'a', 0 }); // one agent "a", seq_base 0
    try evil.append(gpa, 1); // one frame
    try evil.append(gpa, 3); // del run
    try evil.appendSlice(gpa, &.{ 0, 0, 0 }); // agent 0, seq 0, no parents
    // count = 2^40 (LEB128), pos = 0
    try evil.appendSlice(gpa, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x20, 0 });
    try t.expectError(error.Corrupt, victim.merge(gpa, evil.items));
    try t.expectEqual(@as(usize, 0), victim.history.eventCount());
}

// ── Partial bases (holes) ───────────────────────────────────────────────

/// Compact `host` at its current single head and build a partial replica:
/// the base split in three, the middle span left unrealized.
fn partialPair(gpa: std.mem.Allocator, host: *TextDoc, cut1: usize, cut2: usize) !TextDoc {
    const stable = try host.version(gpa);
    defer gpa.free(stable);
    try host.compact(gpa, stable);
    const wm = try host.agentWatermarks(gpa);
    defer gpa.free(wm);
    const base = host.base_bytes;
    const mid_scalars = try std.unicode.utf8CountCodepoints(base[cut1..cut2]);
    return TextDoc.openPartial(gpa, host.base_version, wm, &.{
        .{ .content = base[0..cut1] },
        .{ .hole = .{ .bytes = cut2 - cut1, .scalars = mid_scalars } },
        .{ .content = base[cut2..] },
    });
}

test "partial: sync both ways around the hole; realize completes the doc" {
    const gpa = t.allocator;
    var host: TextDoc = .empty;
    defer host.deinit(gpa);
    try host.setAgent(gpa, "host");
    _ = try host.insert(gpa, 0, "prefix MIDDLE-UNFETCHED suffix");

    var part = try partialPair(gpa, &host, 7, 23);
    defer part.deinit(gpa);
    try t.expect(!part.baseRealized());
    try t.expectEqual(host.text().byteLen(), part.text().byteLen());

    // Versions agree at the base point; sync is a no-op both ways.
    try syncBoth(gpa, &host, &part);

    // Edits in realized regions flow both ways.
    try part.setAgent(gpa, "part");
    _ = try host.insert(gpa, 0, "H:");
    _ = try part.insert(gpa, part.text().byteLen(), ":P");
    try syncBoth(gpa, &host, &part);
    try t.expectEqual(host.text().byteLen(), part.text().byteLen());

    // Partial ops that need base content refuse honestly.
    try t.expectError(error.Unrealized, part.serialize(gpa));
    const ver = try part.version(gpa);
    defer gpa.free(ver);
    try t.expectError(error.Unrealized, part.materializeAt(gpa, ver));

    // Realize the hole (fetch keyed by pristine-base offset) — not an edit.
    const fetch = part.unrealizedBase();
    try t.expectEqual(@as(usize, 1), fetch.len);
    const len_before = part.text().byteLen();
    try part.realizeBase(gpa, fetch[0].base_offset, host.base_bytes[7..23]);
    try t.expect(part.baseRealized());
    try t.expectEqual(len_before, part.text().byteLen());
    try expectConverged(&[_]*TextDoc{ &host, &part });

    // Fully realized: persistence works and round-trips.
    const bytes = try part.serialize(gpa);
    defer gpa.free(bytes);
    var reopened = try TextDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try expectConverged(&[_]*TextDoc{ &part, &reopened });
}

test "partial: edits at hole boundaries stay legal and convergent" {
    const gpa = t.allocator;
    var host: TextDoc = .empty;
    defer host.deinit(gpa);
    try host.setAgent(gpa, "host");
    _ = try host.insert(gpa, 0, "aaaHHHHbbb");
    var part = try partialPair(gpa, &host, 3, 7);
    defer part.deinit(gpa);
    try part.setAgent(gpa, "part");

    // Host edits right at both edges of the region the partial lacks.
    _ = try host.insert(gpa, 3, "[");
    _ = try host.insert(gpa, 8, "]");
    // Partial edits at its own hole edges.
    const h = part.unrealizedBase()[0];
    _ = try part.insert(gpa, h.cur_offset, "(");
    _ = try part.insert(gpa, part.unrealizedBase()[0].cur_offset + part.unrealizedBase()[0].bytes, ")");
    try syncBoth(gpa, &host, &part);
    try t.expectEqual(host.text().byteLen(), part.text().byteLen());

    // Realize and check full convergence (fetch pristine base content).
    const cur = part.unrealizedBase()[0];
    try part.realizeBase(gpa, cur.base_offset, host.base_bytes[3..7]);
    try expectConverged(&[_]*TextDoc{ &host, &part });
}

test "partial: identity anchors are hole-aware — anchor/resolve around a hole names the correct character, unmoved by realize" {
    const gpa = t.allocator;
    var host: TextDoc = .empty;
    defer host.deinit(gpa);
    try host.setAgent(gpa, "host");
    _ = try host.insert(gpa, 0, "aaaHHHHbbb");
    var part = try partialPair(gpa, &host, 3, 7); // hole = "HHHH"
    defer part.deinit(gpa);

    // Real (non-base) edits landing exactly at both edges of the hole —
    // same shape as "partial: edits at hole boundaries stay legal and
    // convergent" above, giving `part` content "aaa[HHHH]bbb" with the
    // middle span still an unrealized hole.
    _ = try host.insert(gpa, 3, "[");
    _ = try host.insert(gpa, 8, "]");
    try syncBoth(gpa, &host, &part);
    try t.expect(!part.baseRealized());

    // `.before` at byte 3 names '[' — right before the hole; needs no
    // hole compensation (a position before every hole is unaffected —
    // included for symmetry/regression, not because it alone proves the
    // fix).
    const at_open = try part.anchorAt(gpa, 3, .before);
    defer gpa.free(at_open.agent);
    // `.before` at byte 8 names ']' — right AFTER the hole. THIS is the
    // review's exact trace: the OLD (realized-only) conversion computed
    // scalar index 4 here (uncompensated: `self.rope.offsetToScalar(8)`
    // alone, ignoring the hole's 4 scalars) — global walker index 4 is
    // the hole's OWN first placeholder item, so the old code reported
    // `error.Compacted` for a position that names a perfectly
    // resolvable REAL character.
    const at_close = try part.anchorAt(gpa, 8, .before);
    defer gpa.free(at_close.agent);

    var offs_pre: [2]usize = undefined;
    try part.resolveAnchors(gpa, &.{ at_open, at_close }, &offs_pre);
    try t.expectEqual(@as(usize, 3), offs_pre[0]);
    try t.expectEqual(@as(usize, 8), offs_pre[1]);

    // Realizing is not an edit: neither anchor moves.
    const h = part.unrealizedBase()[0];
    try part.realizeBase(gpa, h.base_offset, host.base_bytes[3..7]);
    try t.expect(part.baseRealized());
    var offs_post: [2]usize = undefined;
    try part.resolveAnchors(gpa, &.{ at_open, at_close }, &offs_post);
    try t.expectEqualSlices(usize, &offs_pre, &offs_post);

    // Only readable now that the doc is fully realized — check the
    // actual CHARACTER, not just the offset.
    const text = try part.text().toOwnedSlice(gpa);
    defer gpa.free(text);
    try t.expectEqualStrings("aaa[HHHH]bbb", text);
    try t.expectEqual(@as(u8, '['), text[offs_post[0]]);
    try t.expectEqual(@as(u8, ']'), text[offs_post[1]]);
}

test "partial: remote edit into the hole rejects whole, realize-then-merge succeeds" {
    const gpa = t.allocator;
    var host: TextDoc = .empty;
    defer host.deinit(gpa);
    try host.setAgent(gpa, "host");
    _ = try host.insert(gpa, 0, "aaaHHHHbbb");
    var part = try partialPair(gpa, &host, 3, 7);
    defer part.deinit(gpa);

    // Host edits INSIDE the region the partial has not fetched.
    _ = try host.insert(gpa, 5, "xx");
    _ = try host.delete(gpa, .{ .start = 4, .end = 5 });

    const pv = try part.version(gpa);
    defer gpa.free(pv);
    const batch = try host.eventsSince(gpa, pv);
    defer gpa.free(batch);

    const ver_before = try part.version(gpa);
    defer gpa.free(ver_before);
    const len_before = part.text().byteLen();
    try t.expectError(error.Unrealized, part.merge(gpa, batch));
    // Whole-batch rollback: version and text untouched.
    const ver_after = try part.version(gpa);
    defer gpa.free(ver_after);
    try t.expectEqualSlices(u8, ver_before, ver_after);
    try t.expectEqual(len_before, part.text().byteLen());

    // Realize the hole, merge the same batch, converge.
    const h = part.unrealizedBase()[0];
    try part.realizeBase(gpa, h.base_offset, host.base_bytes[3..7]);
    gpa.free(try part.merge(gpa, batch));
    try expectConverged(&[_]*TextDoc{ &host, &part });
}

test "partial: realizeBase validates content; version-only batches cannot bootstrap" {
    const gpa = t.allocator;
    var host: TextDoc = .empty;
    defer host.deinit(gpa);
    try host.setAgent(gpa, "host");
    _ = try host.insert(gpa, 0, "aaaHHHHbbb");
    var part = try partialPair(gpa, &host, 3, 7);
    defer part.deinit(gpa);
    try part.setAgent(gpa, "part");
    _ = try part.insert(gpa, 0, "x"); // so eventsSince has an event to ship

    const h = part.unrealizedBase()[0];
    // Wrong length / bad UTF-8 / wrong scalar count / wrong key.
    try t.expectError(error.Corrupt, part.realizeBase(gpa, h.base_offset, "toolong content"));
    try t.expectError(error.Corrupt, part.realizeBase(gpa, h.base_offset, "\xff\xff\xff\xff"));
    try t.expectError(error.Corrupt, part.realizeBase(gpa, h.base_offset + 1, "HHHH"));
    try t.expect(!part.baseRealized());

    // A batch from the partial carries a version-only base: same-base
    // peers merge it, an empty doc cannot bootstrap from it.
    const hv = try host.version(gpa);
    defer gpa.free(hv);
    const batch = try part.eventsSince(gpa, hv);
    defer gpa.free(batch);
    gpa.free(try host.merge(gpa, batch)); // same base: fine
    var blank: TextDoc = .empty;
    defer blank.deinit(gpa);
    try t.expectError(error.MissingDependency, blank.merge(gpa, batch));
    // And .unit cannot be served from a partial doc at all.
    try t.expectError(error.Unrealized, part.eventsSinceFormat(gpa, hv, .unit));
}

test "eventsBetween: bounded slice syncs a mirror to a past version" {
    const gpa = t.allocator;
    var d: TextDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    _ = try d.insert(gpa, 0, "saved state");
    const saved = try d.version(gpa);
    defer gpa.free(saved);
    _ = try d.insert(gpa, d.text().byteLen(), " and unsaved typing");

    // Mirror follows to the saved point only.
    var mirror: TextDoc = .empty;
    defer mirror.deinit(gpa);
    const mv = try mirror.version(gpa);
    defer gpa.free(mv);
    const slice = try d.eventsBetween(gpa, mv, saved);
    defer gpa.free(slice);
    gpa.free(try mirror.merge(gpa, slice));
    try expectDocText(&mirror, "saved state");

    // Later: catch the mirror up from the saved point to head.
    const mv2 = try mirror.version(gpa);
    defer gpa.free(mv2);
    const rest = try d.eventsSince(gpa, mv2);
    defer gpa.free(rest);
    gpa.free(try mirror.merge(gpa, rest));
    try expectConverged(&[_]*TextDoc{ &d, &mirror });

    // Unknown `to` entries are a hard error, not a guess.
    var stranger: TextDoc = .empty;
    defer stranger.deinit(gpa);
    try stranger.setAgent(gpa, "stranger");
    _ = try stranger.insert(gpa, 0, "x");
    const sv = try stranger.version(gpa);
    defer gpa.free(sv);
    try t.expectError(error.MissingDependency, d.eventsBetween(gpa, mv, sv));
}

test "openFromContent: bulk load, shared root across independent loads" {
    const gpa = t.allocator;
    var a = try TextDoc.openFromContent(gpa, "the same big file contents\n");
    defer a.deinit(gpa);
    var b = try TextDoc.openFromContent(gpa, "the same big file contents\n");
    defer b.deinit(gpa);
    try t.expectEqual(@as(usize, 0), a.history.eventCount());
    try expectDocText(&a, "the same big file contents\n");

    // Independent loads of identical bytes share the history root: they
    // sync as replicas of one document.
    try a.setAgent(gpa, "alice");
    try b.setAgent(gpa, "bob");
    _ = try a.insert(gpa, 0, "A");
    _ = try b.insert(gpa, b.text().byteLen(), "B");
    try syncBoth(gpa, &a, &b);
    try expectConverged(&[_]*TextDoc{ &a, &b });

    // Different contents produce different bases: never confusable.
    var c = try TextDoc.openFromContent(gpa, "entirely different\n");
    defer c.deinit(gpa);
    try c.setAgent(gpa, "carol");
    _ = try c.insert(gpa, 0, "C");
    const av = try a.version(gpa);
    defer gpa.free(av);
    const batch = try c.eventsSince(gpa, av);
    defer gpa.free(batch);
    try t.expectError(error.MissingDependency, a.merge(gpa, batch));
}
