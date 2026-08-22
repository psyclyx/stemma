//! Convergence oracle for ObjectDoc: maps (with honest MV conflicts), lists,
//! nested text objects, wire round-trips, and gossip — replicas that have
//! seen the same events must render byte-identical canonical JSON.

const std = @import("std");
const t = std.testing;

const ObjectDoc = @import("ObjectDoc.zig");
const geometry = @import("../geometry.zig");
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

test "ref: resolves an ObjId returned by a mutator back to a navigable ValueRef" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const tags = (try d.mapSet(gpa, null, "tags", .list)).?;
    _ = try d.listInsert(gpa, tags, 0, .{ .str = "a" });
    // `tags` came straight back from `mapSet`, never touched `root()` — `ref`
    // is the only way to read through it without re-navigating from root.
    try t.expectEqual(ObjectDoc.Kind.list, d.ref(tags).kind());
    try t.expectEqual(@as(usize, 1), d.ref(tags).listLen());
    try t.expectEqualStrings("a", d.ref(tags).listAt(0).asStr());

    // `null` still resolves to the root map, same as `root()`.
    try t.expectEqual(ObjectDoc.Kind.map, d.ref(null).kind());
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

// ── Identity anchors (delta 3: stemma-unification.md §3 step 3) ─────
// Mirrors TextDoc's own anchor battery (text_tests.zig, "identity
// anchors: survive concurrent merges across replicas") one object at a
// time, plus what's genuinely new for a TREE of objects sharing one
// graph: cross-object isolation and portability across a fresh,
// wire-bootstrapped replica.

fn bodyText(gpa: std.mem.Allocator, d: *const ObjectDoc, body: ObjectDoc.ObjId) ![]u8 {
    return d.ref(body).textRope().toOwnedSlice(gpa);
}

test "object anchors: survive concurrent merges across replicas, track the character not the offset" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const body = (try alice.mapSet(gpa, null, "body", .text)).?;
    _ = try alice.textInsert(gpa, body, 0, "hello World");
    try syncBoth(gpa, &alice, &bob);
    const bob_body = bob.root().mapGet("body").?.objId().?;

    // Alice anchors before the 'W'.
    const a = try alice.objectAnchorAt(gpa, body, 6, .before);
    defer gpa.free(a.agent);

    // Divergent concurrent edits on both sides, ahead of AND behind 'W'.
    _ = try alice.textInsert(gpa, body, 0, ">>> ");
    _ = try bob.textInsert(gpa, bob_body, 5, ", cruel");
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });

    // Both replicas resolve the SAME anchor (portable: name+seq) and land
    // on 'W', wherever it now lives in each doc.
    for ([_]struct { d: *ObjectDoc, o: ObjectDoc.ObjId }{
        .{ .d = &alice, .o = body },
        .{ .d = &bob, .o = bob_body },
    }) |c| {
        var off: [1]usize = undefined;
        try c.d.resolveObjectAnchors(gpa, c.o, &.{a}, &off);
        const txt = try bodyText(gpa, c.d, c.o);
        defer gpa.free(txt);
        try t.expectEqual(@as(u8, 'W'), txt[off[0]]);
    }
}

test "object anchors: deleting the target collapses the anchor to the deletion point" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "hello World");
    const a = try d.objectAnchorAt(gpa, body, 6, .before); // before 'W'
    defer gpa.free(a.agent);

    var off_before: [1]usize = undefined;
    try d.resolveObjectAnchors(gpa, body, &.{a}, &off_before);
    _ = try d.textDelete(gpa, body, .{ .start = off_before[0], .end = off_before[0] + 5 }); // "World"
    var off_after: [1]usize = undefined;
    try d.resolveObjectAnchors(gpa, body, &.{a}, &off_after);
    try t.expectEqual(off_before[0], off_after[0]);
}

test "object anchors: boundary anchors resolve to the object's start/end" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "hello");
    const start = try d.objectAnchorAt(gpa, body, 0, .after);
    const end = try d.objectAnchorAt(gpa, body, d.ref(body).textRope().byteLen(), .before);
    var offs: [2]usize = undefined;
    try d.resolveObjectAnchors(gpa, body, &.{ start, end }, &offs);
    try t.expectEqual(@as(usize, 0), offs[0]);
    try t.expectEqual(d.ref(body).textRope().byteLen(), offs[1]);
}

test "object anchors: stickiness sides diverge exactly at the insertion point" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "ac"); // anchor point sits between 'a' and 'c'
    const before = try d.objectAnchorAt(gpa, body, 1, .before); // rides with 'c'
    defer gpa.free(before.agent);
    const after = try d.objectAnchorAt(gpa, body, 1, .after); // rides with 'a'
    defer gpa.free(after.agent);

    // Insert 'b' exactly at the anchor point: "ac" -> "abc".
    _ = try d.textInsert(gpa, body, 1, "b");
    var offs: [2]usize = undefined;
    try d.resolveObjectAnchors(gpa, body, &.{ before, after }, &offs);
    const txt = try bodyText(gpa, &d, body);
    defer gpa.free(txt);
    // `before` stayed attached to 'c' (pushed to index 2 by the insert);
    // `after` stayed attached to 'a' (untouched, still index 1).
    try t.expectEqual(@as(u8, 'c'), txt[offs[0]]);
    try t.expectEqual(@as(usize, 1), offs[1]);
}

test "object anchors: cross-object isolation — heavy concurrent edits to a sibling object don't perturb an anchor" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const body_a = (try alice.mapSet(gpa, null, "a", .text)).?;
    const body_b = (try alice.mapSet(gpa, null, "b", .text)).?;
    _ = try alice.textInsert(gpa, body_a, 0, "target text");
    try syncBoth(gpa, &alice, &bob);
    const bob_b = bob.root().mapGet("b").?.objId().?;

    // Anchor into A, before the 't' of "text" (offset 7).
    const anchor = try alice.objectAnchorAt(gpa, body_a, 7, .before);
    defer gpa.free(anchor.agent);
    var off_before: [1]usize = undefined;
    try alice.resolveObjectAnchors(gpa, body_a, &.{anchor}, &off_before);

    // Heavy, purely concurrent editing of the SIBLING object B, from both
    // replicas, sharing the SAME event graph as A.
    for (0..20) |i| {
        var buf: [8]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "{d}", .{i}) catch unreachable;
        _ = try alice.textInsert(gpa, body_b, 0, s);
        _ = try bob.textInsert(gpa, bob_b, 0, s);
    }
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });

    // A is completely untouched: same content, anchor resolves to the
    // exact same offset it did before any of B's edits happened.
    var off_after: [1]usize = undefined;
    try alice.resolveObjectAnchors(gpa, body_a, &.{anchor}, &off_after);
    try t.expectEqual(off_before[0], off_after[0]);
    const txt = try bodyText(gpa, &alice, body_a);
    defer gpa.free(txt);
    try t.expectEqualStrings("target text", txt);
    try t.expectEqual(@as(u8, 't'), txt[off_after[0]]);
}

test "object anchors: portable across a fresh, wire-bootstrapped replica" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    try alice.setAgent(gpa, "alice");

    const body = (try alice.mapSet(gpa, null, "body", .text)).?;
    _ = try alice.textInsert(gpa, body, 0, "hello World");
    const a = try alice.objectAnchorAt(gpa, body, 6, .before); // before 'W'
    defer gpa.free(a.agent);

    // A brand-new replica, seeded ONLY from wire bytes — never merged
    // incrementally, never shared in-process state with alice.
    const bytes = try alice.serialize(gpa);
    defer gpa.free(bytes);
    var fresh = try ObjectDoc.open(gpa, bytes);
    defer fresh.deinit(gpa);
    const fresh_body = fresh.root().mapGet("body").?.objId().?;

    // The anchor (agent name + seq) resolves correctly purely from the
    // portable identity — no doc-local id ever crossed the wire.
    var off: [1]usize = undefined;
    try fresh.resolveObjectAnchors(gpa, fresh_body, &.{a}, &off);
    const txt = try bodyText(gpa, &fresh, fresh_body);
    defer gpa.free(txt);
    try t.expectEqual(@as(u8, 'W'), txt[off[0]]);
}

test "object anchors: an anchor naming a non-text-insert event is rejected as Corrupt" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const body_a = (try d.mapSet(gpa, null, "a", .text)).?;
    const body_b = (try d.mapSet(gpa, null, "b", .text)).?;
    _ = try d.textInsert(gpa, body_a, 0, "aaa");
    _ = try d.textInsert(gpa, body_b, 0, "bbb");

    // The map_set that created "a" itself: not a text insert at all.
    const map_set_anchor: ObjectDoc.EventAnchor = .{ .agent = "solo", .seq = 0, .side = .before };
    var out: [1]usize = undefined;
    try t.expectError(error.Corrupt, d.resolveObjectAnchors(gpa, body_a, &.{map_set_anchor}, &out));

    // A real text insert, but into the WRONG object (b's first char,
    // resolved against a).
    const wrong_object_anchor = try d.objectAnchorAt(gpa, body_b, 1, .before);
    defer gpa.free(wrong_object_anchor.agent);
    try t.expectError(error.Corrupt, d.resolveObjectAnchors(gpa, body_a, &.{wrong_object_anchor}, &out));
}

test "object anchors: resolving an unknown agent's anchor is MissingDependency" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "hi");

    const ghost: ObjectDoc.EventAnchor = .{ .agent = "ghost", .seq = 0, .side = .before };
    var out: [1]usize = undefined;
    try t.expectError(error.MissingDependency, d.resolveObjectAnchors(gpa, body, &.{ghost}, &out));
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

// ── Compaction (delta 2, stemma-unification.md §3 step 4) ──────────────
// Whole-doc, one linearization point. Object-creating `map_set`/`list_ins`
// events (and, for lists, ANY `list_ins`/`list_del`) are never compacted —
// see `ObjectDoc.compact`'s doc comment for why, and for the per-agent
// PREFIX requirement `causal.compactGraph`'s watermark imposes: a given
// agent's own retained (never-compacted) events must all come AFTER all of
// that same agent's compacted ones. Every scenario below is built to
// respect this on purpose (a dedicated "founder" agent creates structure
// and does nothing else; other agents do all the compactable editing) —
// this is a REAL constraint on real usage, not just test plumbing, so
// exercising it honestly here (rather than routing around it) is part of
// the coverage. `ObjId`s are also doc-local: every cross-replica use below
// re-resolves via `root().mapGet(...).?.objId()` on the RECEIVING replica.

test "compact: text history collapses, map/text content unchanged, collaboration (incl. lists) continues" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    _ = try founder.mapSet(gpa, null, "title", .{ .str = "doc" }); // compactable
    _ = try founder.mapSet(gpa, null, "body", .text); // retained (creation)
    // founder never touches the document again. No list content yet —
    // ANY list_ins/list_del in the causal past of the stable point blocks
    // compaction entirely (see `ObjectDoc.compact`'s doc comment); lists
    // get created AFTER the stable point instead, below.

    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try bob.setAgent(gpa, "bob");
    try syncOne(gpa, &founder, &alice);
    try syncOne(gpa, &founder, &bob);

    const alice_body = alice.root().mapGet("body").?.objId().?;
    _ = try alice.textInsert(gpa, alice_body, 0, "hello world");
    try syncOne(gpa, &alice, &founder);
    try syncOne(gpa, &alice, &bob);

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    const before = try founder.toJson(gpa);
    defer gpa.free(before);
    const events_before = founder.history.eventCount();

    try founder.compact(gpa, stable);
    try bob.compact(gpa, stable);
    // Text content compacted (the doc's bulk); creation events remain, so
    // this shrinks but does not zero out, unlike TextDoc's all-text case.
    try t.expect(founder.history.eventCount() < events_before);
    try t.expect(founder.history.eventCount() > 0);
    const after = try founder.toJson(gpa);
    defer gpa.free(after);
    try t.expectEqualStrings(before, after);
    try expectConverged(&[_]*ObjectDoc{ &founder, &bob });

    // Editing (text AND structural — a list created AFTER the compaction
    // point) and syncing continue across the shared compaction point.
    const bob_body = bob.root().mapGet("body").?.objId().?;
    _ = try bob.textInsert(gpa, bob_body, 0, "A");
    const bob_tags = (try bob.mapSet(gpa, null, "tags", .list)).?;
    _ = try bob.listInsert(gpa, bob_tags, 0, .{ .str = "b" });
    try syncOne(gpa, &bob, &founder);
    try expectConverged(&[_]*ObjectDoc{ &founder, &bob });
}

test "compact: mid-history point keeps later events working" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;

    var carol: ObjectDoc = .empty;
    defer carol.deinit(gpa);
    try carol.setAgent(gpa, "carol");
    try syncOne(gpa, &founder, &carol);
    const carol_body = carol.root().mapGet("body").?.objId().?;
    _ = try carol.textInsert(gpa, carol_body, 0, "early work");
    try syncOne(gpa, &carol, &founder);

    const mid = try founder.version(gpa);
    defer gpa.free(mid);

    _ = try carol.textInsert(gpa, carol_body, carol.ref(carol_body).textRope().byteLen(), " later work");
    _ = try carol.textDelete(gpa, carol_body, .{ .start = 0, .end = 6 });
    try syncOne(gpa, &carol, &founder);

    const text_now = try bodyText(gpa, &founder, body);
    defer gpa.free(text_now);

    try founder.compact(gpa, mid);
    try t.expect(founder.history.eventCount() > 0); // later events retained
    const text_after = try bodyText(gpa, &founder, body);
    defer gpa.free(text_after);
    try t.expectEqualStrings(text_now, text_after);
}

test "compact: rejects multiple heads" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.mapSet(gpa, null, "a", .{ .int = 1 });
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.mapSet(gpa, null, "b", .{ .int = 2 });
    _ = try bob.mapSet(gpa, null, "c", .{ .int = 3 });
    try syncBoth(gpa, &alice, &bob);

    const two_heads = try alice.version(gpa);
    defer gpa.free(two_heads);
    try t.expectError(error.NotCompactable, alice.compact(gpa, two_heads));

    // Doc still fully functional after the rejection.
    _ = try alice.mapSet(gpa, null, "d", .{ .int = 4 });
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });
}

test "compact: rejects a single head that still leaves a concurrent remainder" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    _ = try alice.mapSet(gpa, null, "a", .{ .int = 1 });
    try syncBoth(gpa, &alice, &bob);
    _ = try alice.mapSet(gpa, null, "b", .{ .int = 2 }); // the later, would-be-stable head
    const stale_head = try alice.version(gpa);
    defer gpa.free(stale_head);
    // Concurrent to "b": bob's parent is only "a".
    _ = try bob.mapSet(gpa, null, "c", .{ .int = 3 });
    try syncBoth(gpa, &alice, &bob);

    try t.expectError(error.NotCompactable, alice.compact(gpa, stale_head));
    _ = try alice.mapSet(gpa, null, "d", .{ .int = 4 });
    try syncBoth(gpa, &alice, &bob);
    try expectConverged(&[_]*ObjectDoc{ &alice, &bob });
}

test "compact: rejects a stable point whose causal past still has list structure" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const tags = (try founder.mapSet(gpa, null, "tags", .list)).?;
    _ = try founder.listInsert(gpa, tags, 0, .{ .str = "a" });
    const stable = try founder.version(gpa);
    defer gpa.free(stable);

    try t.expectError(error.NotCompactable, founder.compact(gpa, stable));
    // Doc still fully functional after the rejection.
    _ = try founder.listInsert(gpa, tags, 1, .{ .str = "b" });
    try t.expectEqual(@as(usize, 2), founder.root().mapGet("tags").?.listLen());
}

test "compact: identity anchors into compacted text become error.Compacted" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;

    var carol: ObjectDoc = .empty;
    defer carol.deinit(gpa);
    try carol.setAgent(gpa, "carol");
    try syncOne(gpa, &founder, &carol);
    const carol_body = carol.root().mapGet("body").?.objId().?;
    _ = try carol.textInsert(gpa, carol_body, 0, "hello");
    try syncOne(gpa, &carol, &founder);

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    try founder.compact(gpa, stable);

    try t.expectError(error.Compacted, founder.objectAnchorAt(gpa, body, 3, .before));

    // Post-compaction content still anchors fine.
    _ = try founder.textInsert(gpa, body, 5, "!");
    const a = try founder.objectAnchorAt(gpa, body, 5, .before);
    defer gpa.free(a.agent);
    var off: [1]usize = undefined;
    try founder.resolveObjectAnchors(gpa, body, &.{a}, &off);
    try t.expectEqual(@as(usize, 5), off[0]);
}

test "compact: serialize/open round-trip preserves compacted content and keeps collaborating" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    _ = try founder.mapSet(gpa, null, "title", .{ .str = "doc" });
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;

    var carol: ObjectDoc = .empty;
    defer carol.deinit(gpa);
    try carol.setAgent(gpa, "carol");
    try syncOne(gpa, &founder, &carol);
    const carol_body = carol.root().mapGet("body").?.objId().?;
    _ = try carol.textInsert(gpa, carol_body, 0, "hello world");
    try syncOne(gpa, &carol, &founder);

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    try founder.compact(gpa, stable);

    const bytes = try founder.serialize(gpa);
    defer gpa.free(bytes);
    var reopened = try ObjectDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try expectConverged(&[_]*ObjectDoc{ &founder, &reopened });
    _ = body;

    try reopened.setAgent(gpa, "editor");
    const reopened_body = reopened.root().mapGet("body").?.objId().?;
    _ = try reopened.textInsert(gpa, reopened_body, 0, "X");
    try syncOne(gpa, &reopened, &founder);
    try expectConverged(&[_]*ObjectDoc{ &founder, &reopened });
}

test "resume sync after compact: uncompacted peer is rejected until it compacts to the same point, then converges" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;

    var carol: ObjectDoc = .empty;
    defer carol.deinit(gpa);
    try carol.setAgent(gpa, "carol");
    try syncOne(gpa, &founder, &carol);
    const carol_body = carol.root().mapGet("body").?.objId().?;
    _ = try carol.textInsert(gpa, carol_body, 0, "shared history");
    try syncOne(gpa, &carol, &founder);

    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try bob.setAgent(gpa, "bob");
    try syncOne(gpa, &founder, &bob); // bob catches up fully BEFORE compaction

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    try founder.compact(gpa, stable);
    _ = try founder.textInsert(gpa, body, 0, "x");

    // Bob is causally caught up to the stable point, but has NOT compacted
    // locally — still rejected, the loud documented behavior (mirrors
    // `TextDoc`'s "uncompacted peer cannot merge a based batch").
    const vb = try bob.version(gpa);
    defer gpa.free(vb);
    const batch = try founder.eventsSince(gpa, vb);
    defer gpa.free(batch);
    try t.expectError(error.MissingDependency, bob.merge(gpa, batch));

    // After compacting to the same point, sync works again.
    try bob.compact(gpa, stable);
    try syncOne(gpa, &founder, &bob);
    try expectConverged(&[_]*ObjectDoc{ &founder, &bob });
}

test "compact: map register conflict resolution stays correct for a late concurrent peer" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;
    _ = body;

    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try bob.setAgent(gpa, "bob");
    try syncOne(gpa, &founder, &alice);
    try syncOne(gpa, &founder, &bob);

    // Alice edits body's text; founder takes its stable point RIGHT THERE
    // — before alice's map write to "k" — so `s` is alice's own last
    // (foldable) `text_ins`, and every retained event's `in_base` parent
    // is exactly `s` (see "compact: refuses when a retained write's only
    // path to another retained write is through a folded text edit" for
    // what goes wrong if a retained write's causal path to `s` — or to
    // another retained write — crosses a folded edit instead).
    const alice_body = alice.root().mapGet("body").?.objId().?;
    _ = try alice.textInsert(gpa, alice_body, 0, "hello world");
    try syncOne(gpa, &alice, &founder);

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    try founder.compact(gpa, stable);

    // Alice's write to "k" (retained: map ops never compact) and bob's
    // concurrent one both happen AFTER the stable point — alice's parent
    // resolves through founder's `base_head` (the implicit-base-parent
    // path `merge` already handles for a non-based, ordinary batch from a
    // peer who's caught up to the compaction point).
    _ = try alice.mapSet(gpa, null, "k", .{ .str = "v1" });
    _ = try bob.mapSet(gpa, null, "k", .{ .str = "v2" });
    try syncOne(gpa, &alice, &founder);

    // Bob's still-unsynced, causally-concurrent write to "k" arrives AFTER
    // compaction.
    const vf = try founder.version(gpa);
    defer gpa.free(vf);
    const batch = try bob.eventsSince(gpa, vf);
    defer gpa.free(batch);
    const changes = try founder.merge(gpa, batch);
    gpa.free(changes);

    try t.expectEqual(@as(usize, 2), founder.root().mapConflictCount("k"));
    var saw_v1 = false;
    var saw_v2 = false;
    for (0..2) |i| {
        const v = founder.root().mapConflictAt("k", i).asStr();
        if (std.mem.eql(u8, v, "v1")) saw_v1 = true;
        if (std.mem.eql(u8, v, "v2")) saw_v2 = true;
    }
    try t.expect(saw_v1 and saw_v2);
}

// Regression coverage for a real, review-caught soundness bug: the FIRST
// version of `ObjectDoc.compact` accepted a stable point whose only
// retained event proving Q-supersedes-P routed through a folded text
// edit — `causal.compactGraph`'s blanket "drop any in_base parent" then
// silently dropped that edge, and a fresh replica reopening the SAME
// compacted bytes computed a DIFFERENT winner for "k" than the compacting
// replica had (proven empirically before the fix: conflictCount and the
// agent-name tiebreak both diverged). The fix is refusal, not repair —
// `ObjectDoc.compact` now rejects this shape outright
// (`error.NotCompactable`); see its doc comment and
// `causal.compactGraph`'s own defensive `assert` for the two lines of
// defense.

test "compact: refuses when a retained write's only path to another retained write is through a folded text edit" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    _ = try founder.mapSet(gpa, null, "body", .text);

    var zzz: ObjectDoc = .empty;
    defer zzz.deinit(gpa);
    try zzz.setAgent(gpa, "zzz");
    var aaa: ObjectDoc = .empty;
    defer aaa.deinit(gpa);
    try aaa.setAgent(gpa, "aaa");
    try syncOne(gpa, &founder, &zzz);

    // P: zzz writes "k" first.
    _ = try zzz.mapSet(gpa, null, "k", .{ .str = "P-zzz" });
    // T: zzz then edits body's text — zzz's own local frontier chains
    // P -> T directly, so T is the sole causal link out of P for anyone
    // who only saw zzz's history up through T.
    const zzz_body = zzz.root().mapGet("body").?.objId().?;
    _ = try zzz.textInsert(gpa, zzz_body, 0, "hi");

    // aaa catches up to T (and, transitively, P) — her own frontier is
    // exactly {T}, never {P} directly.
    try syncOne(gpa, &zzz, &aaa);
    // Q: aaa's write supersedes P — she has SEEN it (via T).
    _ = try aaa.mapSet(gpa, null, "k", .{ .str = "Q-aaa" });

    try syncOne(gpa, &aaa, &founder);
    try t.expectEqual(@as(usize, 1), founder.root().mapConflictCount("k"));
    try t.expectEqualStrings("Q-aaa", founder.root().mapGet("k").?.asStr());

    // The stable point is Q itself: Q's ONLY causal path back to P routes
    // through the now-foldable T. Compacting here would make
    // `causal.compactGraph` drop T's edge (T is in_base, T != s == Q),
    // silently losing Q's dependency on P — refused instead.
    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    try t.expectError(error.NotCompactable, founder.compact(gpa, stable));

    // The refusal is a no-op, not a partial mutation: content is exactly
    // as it was, and stays correct across a real merge/serialize round
    // trip too (uncompacted, since compaction never happened).
    try t.expectEqual(@as(usize, 1), founder.root().mapConflictCount("k"));
    try t.expectEqualStrings("Q-aaa", founder.root().mapGet("k").?.asStr());
    const bytes = try founder.serialize(gpa);
    defer gpa.free(bytes);
    var reopened = try ObjectDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try t.expectEqual(@as(usize, 1), reopened.root().mapConflictCount("k"));
    try t.expectEqualStrings("Q-aaa", reopened.root().mapGet("k").?.asStr());
}

test "compact: still succeeds when a retained write's path to another retained write doesn't cross a folded edit" {
    const gpa = t.allocator;
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    _ = try founder.mapSet(gpa, null, "body", .text);
    // founder never touches the document again — same reason as every
    // other test in this file (the per-agent PREFIX constraint:
    // `causal.compactGraph`'s watermark needs a given agent's compactable
    // events to be their EARLIEST, never interleaved with retained ones —
    // see `ObjectDoc.compact`'s doc comment).

    var zzz: ObjectDoc = .empty;
    defer zzz.deinit(gpa);
    try zzz.setAgent(gpa, "zzz");
    var aaa: ObjectDoc = .empty;
    defer aaa.deinit(gpa);
    try aaa.setAgent(gpa, "aaa");
    var carol: ObjectDoc = .empty;
    defer carol.deinit(gpa);
    try carol.setAgent(gpa, "carol");
    try syncOne(gpa, &founder, &zzz);

    // Same P/Q shape as the regression test above, but WITHOUT a text
    // edit sitting between them: aaa's frontier is exactly {P} when she
    // writes Q, a direct edge.
    _ = try zzz.mapSet(gpa, null, "k", .{ .str = "P-zzz" });
    try syncOne(gpa, &zzz, &aaa);
    _ = try aaa.mapSet(gpa, null, "k", .{ .str = "Q-aaa" });
    try syncOne(gpa, &aaa, &founder);

    // Unrelated text content from a THIRD agent (carol), whose own
    // timeline is entirely compactable text edits — no retained events of
    // her own to interleave with, so she can't violate the per-agent
    // prefix either. Synced from founder AFTER Q, so her edit's frontier
    // includes Q (no concurrent second head).
    try syncOne(gpa, &founder, &carol);
    const carol_body = carol.root().mapGet("body").?.objId().?;
    _ = try carol.textInsert(gpa, carol_body, 0, "hello");
    try syncOne(gpa, &carol, &founder);

    try t.expectEqual(@as(usize, 1), founder.root().mapConflictCount("k"));
    try t.expectEqualStrings("Q-aaa", founder.root().mapGet("k").?.asStr());

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    try founder.compact(gpa, stable);

    try t.expectEqual(@as(usize, 1), founder.root().mapConflictCount("k"));
    try t.expectEqualStrings("Q-aaa", founder.root().mapGet("k").?.asStr());
}

fn compactOomScript(gpa: std.mem.Allocator) !void {
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;
    _ = body;

    var carol: ObjectDoc = .empty;
    defer carol.deinit(gpa);
    try carol.setAgent(gpa, "carol");
    try syncOne(gpa, &founder, &carol);
    const carol_body = carol.root().mapGet("body").?.objId().?;
    _ = try carol.textInsert(gpa, carol_body, 0, "compact under pressure");
    try syncOne(gpa, &carol, &founder);

    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    _ = try carol.textInsert(gpa, carol_body, 0, "x");
    try syncOne(gpa, &carol, &founder);

    try founder.compact(gpa, stable);
    const bytes = try founder.serialize(gpa);
    defer gpa.free(bytes);
    var re = try ObjectDoc.open(gpa, bytes);
    defer re.deinit(gpa);
}

test "OOM: compaction paths are leak-free under every allocation failure" {
    try std.testing.checkAllAllocationFailures(t.allocator, compactOomScript, .{});
}
