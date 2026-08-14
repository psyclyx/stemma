//! Convergence oracle and hardening tests for the collaboration layer.
//! The load-bearing property is CONVERGENCE: any set of replicas that have
//! seen the same events materialize byte-identical documents, regardless of
//! merge order, batch splitting, or gossip topology. Secondary gates: the
//! returned edit streams are valid (in-bounds, length-consistent — they are
//! the caller's anchor-shifting input), the wire format round-trips, and
//! nothing leaks under allocation failure.

const std = @import("std");
const t = std.testing;

const doc_mod = @import("doc.zig");
const geometry = @import("../geometry.zig");
const anchors_mod = @import("../anchors.zig");

const TextDoc = doc_mod.TextDoc;
const EventId = doc_mod.EventId;
const Range = geometry.Range;
const Edit = geometry.Edit;
const AnchorSet = anchors_mod.AnchorSet;

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
