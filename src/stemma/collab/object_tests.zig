//! Convergence oracle for ObjectDoc: maps (with honest MV conflicts), lists,
//! nested text objects, wire round-trips, and gossip — replicas that have
//! seen the same events must render byte-identical canonical JSON.

const std = @import("std");
const t = std.testing;

const json = @import("objects.zig");
const geometry = @import("../geometry.zig");
const ObjectDoc = json.ObjectDoc;
const Range = geometry.Range;

fn syncOne(gpa: std.mem.Allocator, from: *const ObjectDoc, to: *ObjectDoc) !void {
    const ver = try to.version(gpa);
    defer gpa.free(ver);
    const batch = try from.eventsSince(gpa, ver);
    defer gpa.free(batch);
    const changes = try to.merge(gpa, batch);
    gpa.free(changes);
}

fn syncBoth(gpa: std.mem.Allocator, a: *ObjectDoc, b: *ObjectDoc) !void {
    try syncOne(gpa, a, b);
    try syncOne(gpa, b, a);
}

fn expectConverged(docs: []const *ObjectDoc) !void {
    const gpa = t.allocator;
    const first = try docs[0].toJson(gpa);
    defer gpa.free(first);
    for (docs[1..]) |d| {
        const other = try d.toJson(gpa);
        defer gpa.free(other);
        try t.expectEqualStrings(first, other);
    }
}

test "local tree building renders canonical JSON" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    _ = try d.mapSet(gpa, null, "title", .{ .str = "My Doc" });
    _ = try d.mapSet(gpa, null, "count", .{ .int = 42 });
    _ = try d.mapSet(gpa, null, "done", .{ .bool_ = false });
    const meta = (try d.mapSet(gpa, null, "meta", .map)).?;
    _ = try d.mapSet(gpa, meta, "nested", .null_);
    const tags = (try d.mapSet(gpa, null, "tags", .list)).?;
    _ = try d.listInsert(gpa, tags, 0, .{ .str = "a" });
    _ = try d.listInsert(gpa, tags, 1, .{ .str = "c" });
    _ = try d.listInsert(gpa, tags, 1, .{ .str = "b" });
    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "hello 世界");
    _ = try d.textDelete(gpa, body, .{ .start = 0, .end = 6 });
    _ = try d.textInsert(gpa, body, 0, "goodbye ");

    const got = try d.toJson(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings(
        \\{"body":"goodbye 世界","count":42,"done":false,"meta":{"nested":null},"tags":["a","b","c"],"title":"My Doc"}
    , got);

    // Reads.
    const r = d.root();
    try t.expectEqual(@as(i64, 42), r.mapGet("count").?.asInt());
    try t.expectEqual(@as(usize, 3), r.mapGet("tags").?.listLen());
    try t.expectEqualStrings("b", r.mapGet("tags").?.listAt(1).asStr());
    try t.expect(r.mapGet("absent") == null);

    _ = try d.mapDelete(gpa, null, "done");
    try t.expect(d.root().mapGet("done") == null);
}

test "concurrent map sets: both survive as conflicts, winner deterministic" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.mapSet(gpa, null, "status", .{ .str = "draft" });
    try syncBoth(gpa, &alice, &bob);

    // Concurrent overwrites.
    _ = try alice.mapSet(gpa, null, "status", .{ .str = "review" });
    _ = try bob.mapSet(gpa, null, "status", .{ .str = "final" });
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });

    // Honest MV: both values present, same deterministic winner everywhere.
    try t.expectEqual(@as(usize, 2), alice.root().mapConflictCount("status"));
    try t.expectEqual(@as(usize, 2), bob.root().mapConflictCount("status"));
    const wa = alice.root().mapGet("status").?.asStr();
    const wb = bob.root().mapGet("status").?.asStr();
    try t.expectEqualStrings(wa, wb);
    try t.expectEqualStrings("final", wa); // "bob" > "alice" by name

    // A causally-later set collapses the conflict.
    _ = try alice.mapSet(gpa, null, "status", .{ .str = "published" });
    try syncBoth(gpa, &alice, &bob);
    try t.expectEqual(@as(usize, 1), bob.root().mapConflictCount("status"));
    try t.expectEqualStrings("published", bob.root().mapGet("status").?.asStr());
}

test "concurrent set vs delete: the set survives (add wins over absent)" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.mapSet(gpa, null, "k", .{ .int = 1 });
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.mapSet(gpa, null, "k", .{ .int = 2 });
    try bob.mapDelete(gpa, null, "k");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });
    // Bob's delete removed the value he saw (1); alice's concurrent set of 2
    // was not seen by the delete, so it survives.
    try t.expectEqual(@as(i64, 2), bob.root().mapGet("k").?.asInt());
}

test "concurrent list inserts converge without duplication" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const items = (try alice.mapSet(gpa, null, "items", .list)).?;
    _ = try alice.listInsert(gpa, items, 0, .{ .int = 0 });
    try syncBoth(gpa, &alice, &bob);

    // ObjIds are doc-local handles: bob addresses the shared list through
    // HIS document (structure navigation), or via export/import tokens.
    const bob_items = bob.root().mapGet("items").?.objId().?;
    const token = try alice.exportId(gpa, items);
    defer gpa.free(token);
    const imported = try bob.importId(token);
    try t.expectEqual(bob_items.agent, imported.agent);
    try t.expectEqual(bob_items.seq, imported.seq);

    _ = try alice.listInsert(gpa, items, 1, .{ .int = 1 });
    _ = try alice.listInsert(gpa, items, 2, .{ .int = 2 });
    _ = try bob.listInsert(gpa, bob_items, 1, .{ .int = 10 });
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });
    try t.expectEqual(@as(usize, 4), alice.root().mapGet("items").?.listLen());

    // Concurrent deletes of the same element collapse to one removal.
    _ = try alice.listDelete(gpa, items, 0);
    _ = try bob.listDelete(gpa, bob_items, 0);
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });
    try t.expectEqual(@as(usize, 3), alice.root().mapGet("items").?.listLen());
}

test "nested text objects: concurrent editing converges; changes are valid edits" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const body = (try alice.mapSet(gpa, null, "body", .text)).?;
    _ = try alice.textInsert(gpa, body, 0, "shared text");
    try syncBoth(gpa, &alice, &bob);
    const bob_body = bob.root().mapGet("body").?.objId().?;

    _ = try alice.textInsert(gpa, body, 0, "[a] ");
    _ = try bob.textInsert(gpa, bob_body, 11, "!");

    // Validate bob's incoming change stream against his text object length.
    {
        const ver = try bob.version(gpa);
        defer gpa.free(ver);
        const batch = try alice.eventsSince(gpa, ver);
        defer gpa.free(batch);
        var len = bob.root().mapGet("body").?.textRope().byteLen();
        const changes = try bob.merge(gpa, batch);
        defer gpa.free(changes);
        for (changes) |c| {
            try t.expect(c == .text);
            try t.expect(c.text.edit.offset <= len);
            len = len - c.text.edit.removed + c.text.edit.inserted;
        }
        try t.expectEqual(bob.root().mapGet("body").?.textRope().byteLen(), len);
    }
    try syncOne(gpa, &bob, &alice);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });

    const txt = try alice.root().mapGet("body").?.textRope().toOwnedSlice(gpa);
    defer gpa.free(txt);
    try t.expectEqualStrings("[a] shared text!", txt);
}

test "serialize/open roundtrip; compareVersions over json docs" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    _ = try d.mapSet(gpa, null, "x", .{ .float = 1.5 });
    const v1 = try d.version(gpa);
    defer gpa.free(v1);
    const list = (try d.mapSet(gpa, null, "l", .list)).?;
    _ = try d.listInsert(gpa, list, 0, .null_);
    const v2 = try d.version(gpa);
    defer gpa.free(v2);

    try t.expectEqual(.ancestor, try d.compareVersions(gpa, v1, v2));
    try t.expectEqual(.equal, try d.compareVersions(gpa, v2, v2));

    const bytes = try d.serialize(gpa);
    defer gpa.free(bytes);
    var re = try ObjectDoc.open(gpa, bytes);
    defer re.deinit(gpa);
    try expectConverged(&[_]*ObjectDoc{ &d, &re });

    // The reopened doc collaborates.
    try re.setAgent(gpa, "editor");
    _ = try re.mapSet(gpa, null, "x", .{ .float = 2.5 });
    try syncBoth(gpa, &re, &d);
    try expectConverged(&[_]*ObjectDoc{ &d, &re });
}

test "malicious json batches rejected without damage" {
    const gpa = t.allocator;
    var victim: ObjectDoc = .empty;
    defer victim.deinit(gpa);
    try victim.setAgent(gpa, "victim");
    _ = try victim.mapSet(gpa, null, "safe", .{ .int = 1 });
    const before = try victim.toJson(gpa);
    defer gpa.free(before);
    const events_before = victim.history.eventCount();

    try t.expectError(error.Corrupt, victim.merge(gpa, "junk"));

    // Structurally valid batch, list op on the ROOT (type confusion).
    const evil_root_list = "stj\x01" ++ [_]u8{ 1, 4 } ++ "evil" ++
        [_]u8{ 1, 0, 0, 0, 2, 0, 0 }; // list_ins, obj=root
    try t.expectError(error.Corrupt, victim.merge(gpa, evil_root_list));

    // Text op against an object that is a scalar (kind confusion): first a
    // legit map_set of an int by "evil", then a text_ins targeting it.
    const evil_text_on_int = "stj\x01" ++ [_]u8{ 1, 4 } ++ "evil" ++
        [_]u8{2} ++ // two events
        [_]u8{ 0, 0, 0, 0, 0, 1, 'k', 3, 2 } ++ // map_set root "k" int 1
        [_]u8{ 0, 1, 1, 0, 0, 4, 1, 0, 0, 5, 'x' }; // text_ins obj=(evil,0)
    try t.expectError(error.Corrupt, victim.merge(gpa, evil_text_on_int));

    try t.expectEqual(events_before, victim.history.eventCount());
    const after = try victim.toJson(gpa);
    defer gpa.free(after);
    try t.expectEqualStrings(before, after);
}

test "three peers, seeded random gossip over mixed structures, convergence" {
    const gpa = t.allocator;
    var docs: [3]ObjectDoc = .{ .empty, .empty, .empty };
    defer for (&docs) |*d| d.deinit(gpa);
    const names = [_][]const u8{ "alice", "bob", "carol" };
    for (&docs, names) |*d, n| try d.setAgent(gpa, n);

    // Shared scaffolding from one peer; each doc resolves its own handles.
    _ = (try docs[0].mapSet(gpa, null, "list", .list)).?;
    _ = (try docs[0].mapSet(gpa, null, "text", .text)).?;
    try syncOne(gpa, &docs[0], &docs[1]);
    try syncOne(gpa, &docs[0], &docs[2]);

    const keys = [_][]const u8{ "a", "b", "c" };
    for ([_]u64{ 7, 0xabcd }) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (0..30) |_| {
            const d = &docs[random.uintLessThan(usize, docs.len)];
            switch (random.uintLessThan(u8, 5)) {
                0 => _ = try d.mapSet(gpa, null, keys[random.uintLessThan(usize, keys.len)], .{ .int = random.int(i32) }),
                1 => {
                    const l = d.root().mapGet("list").?;
                    _ = try d.listInsert(gpa, l.objId().?, random.uintLessThan(usize, l.listLen() + 1), .{ .int = random.int(i16) });
                },
                2 => {
                    const l = d.root().mapGet("list").?;
                    if (l.listLen() > 0) {
                        try d.listDelete(gpa, l.objId().?, random.uintLessThan(usize, l.listLen()));
                    }
                },
                3 => {
                    const tv = d.root().mapGet("text").?;
                    const r = tv.textRope();
                    const pos = r.scalarToOffset(random.uintLessThan(usize, r.scalarLen() + 1));
                    _ = try d.textInsert(gpa, tv.objId().?, pos, "ab");
                },
                4 => {
                    const i = random.uintLessThan(usize, docs.len);
                    var j = random.uintLessThan(usize, docs.len);
                    if (i == j) j = (j + 1) % docs.len;
                    try syncOne(gpa, &docs[i], &docs[j]);
                },
                else => unreachable,
            }
        }
        for (0..2) |_| {
            for (0..docs.len) |i| {
                for (0..docs.len) |j| {
                    if (i != j) try syncOne(gpa, &docs[i], &docs[j]);
                }
            }
        }
        try expectConverged(&[_]*ObjectDoc{ &docs[0], &docs[1], &docs[2] });
    }
}

fn oomScript(gpa: std.mem.Allocator) !void {
    var a: ObjectDoc = .empty;
    defer a.deinit(gpa);
    var b: ObjectDoc = .empty;
    defer b.deinit(gpa);
    try a.setAgent(gpa, "a");
    try b.setAgent(gpa, "b");
    const txt = (try a.mapSet(gpa, null, "t", .text)).?;
    _ = try a.textInsert(gpa, txt, 0, "hi");
    _ = try a.mapSet(gpa, null, "n", .{ .int = 5 });
    const bytes = try a.serialize(gpa);
    defer gpa.free(bytes);
    gpa.free(try b.merge(gpa, bytes));
    _ = try b.mapSet(gpa, null, "n", .{ .int = 6 });
    const vb = try a.version(gpa);
    defer gpa.free(vb);
    const batch = try b.eventsSince(gpa, vb);
    defer gpa.free(batch);
    gpa.free(try a.merge(gpa, batch));
    const dump = try a.toJson(gpa);
    gpa.free(dump);
}

test "OOM: json collaboration paths are leak-free" {
    try std.testing.checkAllAllocationFailures(t.allocator, oomScript, .{});
}

fn fuzzWire(_: void, smith: *std.testing.Smith) !void {
    const gpa = t.allocator;
    var author: ObjectDoc = .empty;
    defer author.deinit(gpa);
    try author.setAgent(gpa, "author");
    const l = (try author.mapSet(gpa, null, "l", .list)).?;
    _ = try author.listInsert(gpa, l, 0, .{ .str = "s" });
    const txt = (try author.mapSet(gpa, null, "t", .text)).?;
    _ = try author.textInsert(gpa, txt, 0, "wire");
    const valid = try author.serialize(gpa);
    defer gpa.free(valid);

    const mutated = try gpa.dupe(u8, valid);
    defer gpa.free(mutated);
    for (0..1 + smith.indexWithHash(4, 0x0f11)) |_| {
        mutated[smith.indexWithHash(mutated.len, 0x0a7e)] = smith.valueWithHash(u8, 0xb17e);
    }
    var victim: ObjectDoc = .empty;
    defer victim.deinit(gpa);
    if (victim.merge(gpa, mutated)) |changes| {
        gpa.free(changes);
    } else |err| switch (err) {
        error.Corrupt, error.MissingDependency => {
            try t.expectEqual(@as(usize, 0), victim.history.eventCount());
        },
        else => |e| return e,
    }
}

test "fuzz: mutated json wire bytes never crash, rejects are atomic" {
    try std.testing.fuzz({}, fuzzWire, .{});
}
