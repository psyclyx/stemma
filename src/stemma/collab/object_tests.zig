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
    // A well-formed structural op (F3, delta 6 — `OpTag` values 6/7) in
    // the seed doc, so byte mutations below land on GENUINE
    // struct_create/struct_move wire regions (op tag, struct-ref tag,
    // order-key length/bytes) some of the time, rather than needing a
    // mutation to incidentally produce tag 6/7 from scratch out of
    // otherwise-map/list/text bytes.
    const sk = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(sk);
    const sn = try author.structCreate(gpa, .root, sk);
    const sk2 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(sk2);
    try author.structMove(gpa, sn, .trash, sk2);
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

// ── W7-1: eventsBetween, bulk load, materializeAt ───────────────────────
// `app/weft/doc/w7-rebase.md` §1/§4 — the four `Document`-consumed gaps
// `stemma-unification.md` §4 risks 4/5 scoped OUT of the unification
// deltas. This section closes the three that fit cleanly on the existing
// compaction machinery (`base_version`/`text_bases`/`seq_base`); the
// fourth (partial checkout) is reported as a design, not built here.

test "wire compat: an ordinary uncompacted doc's serialize() bytes are byte-identical to before W7-1" {
    // Locks the claim this whole section's doc comments make ("purely
    // additive; no existing wire bytes change shape") against silent
    // drift: none of `eventsBetween`/`openFromContent`/`materializeAt`
    // touch `encodeEvents`/`Decoder` at all, so an everyday v1 doc (never
    // compacted, never bulk-loaded) must still emit exactly what it did
    // before this lane. Golden bytes captured from this exact sequence of
    // ops at HEAD before W7-1 landed.
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");
    _ = try d.mapSet(gpa, null, "title", .{ .str = "hello" });
    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "hi");

    const bytes = try d.serialize(gpa);
    defer gpa.free(bytes);
    const golden = [_]u8{
        0x73, 0x74, 0x6a, 0x01, 0x01, 0x04, 0x73, 0x6f, 0x6c, 0x6f, 0x04, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x05, 0x74, 0x69, 0x74, 0x6c, 0x65, 0x05, 0x05,
        0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x62, 0x6f, 0x64, 0x79, 0x08, 0x00, 0x02, 0x01, 0x00, 0x01, 0x04,
        0x01, 0x00, 0x01, 0x00, 0x68, 0x00, 0x03, 0x01, 0x00, 0x02, 0x04, 0x01,
        0x00, 0x01, 0x01, 0x69,
    };
    try t.expectEqualSlices(u8, &golden, bytes);
    // Also starts with `object_magic_v1` ("stj\x01") — never v2 — since
    // this doc neither compacted nor bulk-loaded.
    try t.expectEqualStrings("stj\x01", bytes[0..4]);
}

test "eventsBetween: bounded slice syncs a mirror to a past version" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    _ = try d.mapSet(gpa, null, "title", .{ .str = "saved" });
    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "saved state");
    const saved = try d.version(gpa);
    defer gpa.free(saved);
    _ = try d.textInsert(gpa, body, d.ref(body).textRope().byteLen(), " and unsaved typing");

    // Mirror follows to the saved point only.
    var mirror: ObjectDoc = .empty;
    defer mirror.deinit(gpa);
    const mv = try mirror.version(gpa);
    defer gpa.free(mv);
    const slice = try d.eventsBetween(gpa, mv, saved);
    defer gpa.free(slice);
    gpa.free(try mirror.merge(gpa, slice));
    const mirror_body = mirror.root().mapGet("body").?.objId().?;
    const mirror_text = try bodyText(gpa, &mirror, mirror_body);
    defer gpa.free(mirror_text);
    try t.expectEqualStrings("saved state", mirror_text);

    // Later: catch the mirror up from the saved point to head.
    const mv2 = try mirror.version(gpa);
    defer gpa.free(mv2);
    const rest = try d.eventsSince(gpa, mv2);
    defer gpa.free(rest);
    gpa.free(try mirror.merge(gpa, rest));
    try expectConverged(&[_]*ObjectDoc{ &d, &mirror });

    // Unknown `to` entries are a hard error, not a guess.
    var stranger: ObjectDoc = .empty;
    defer stranger.deinit(gpa);
    try stranger.setAgent(gpa, "stranger");
    _ = try stranger.mapSet(gpa, null, "x", .{ .int = 1 });
    const sv = try stranger.version(gpa);
    defer gpa.free(sv);
    try t.expectError(error.MissingDependency, d.eventsBetween(gpa, mv, sv));
}

test "eventsBetween: cross-check against eventsSince (from empty == eventsSince)" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    _ = try d.mapSet(gpa, null, "title", .{ .str = "x" });
    const body = (try d.mapSet(gpa, null, "body", .text)).?;
    _ = try d.textInsert(gpa, body, 0, "hello");

    var empty: ObjectDoc = .empty;
    defer empty.deinit(gpa);
    const empty_v = try empty.version(gpa);
    defer gpa.free(empty_v);
    const head_v = try d.version(gpa);
    defer gpa.free(head_v);

    const via_between = try d.eventsBetween(gpa, empty_v, head_v);
    defer gpa.free(via_between);
    const via_since = try d.eventsSince(gpa, empty_v);
    defer gpa.free(via_since);

    var a: ObjectDoc = .empty;
    defer a.deinit(gpa);
    gpa.free(try a.merge(gpa, via_between));
    var b: ObjectDoc = .empty;
    defer b.deinit(gpa);
    gpa.free(try b.merge(gpa, via_since));
    try expectConverged(&[_]*ObjectDoc{ &a, &b, &d });
}

test "eventsBetween: `from` causally ahead of `to` yields an empty batch" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    const early = try d.version(gpa);
    defer gpa.free(early);
    _ = try d.mapSet(gpa, null, "k", .{ .int = 1 });
    const later = try d.version(gpa);
    defer gpa.free(later);

    // `from` = the current head, `to` = an earlier point: nothing new to
    // report going "backwards".
    const batch = try d.eventsBetween(gpa, later, early);
    defer gpa.free(batch);
    var mirror: ObjectDoc = .empty;
    defer mirror.deinit(gpa);
    gpa.free(try mirror.merge(gpa, batch));
    try t.expectEqual(@as(usize, 0), mirror.history.eventCount());
}

test "eventsBetween: `from` naming an agent we've never heard of is ignored (lenient), not an error" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    _ = try d.mapSet(gpa, null, "k", .{ .int = 1 });
    const to = try d.version(gpa);
    defer gpa.free(to);

    var stranger: ObjectDoc = .empty;
    defer stranger.deinit(gpa);
    try stranger.setAgent(gpa, "stranger");
    _ = try stranger.mapSet(gpa, null, "x", .{ .int = 9 });
    const stranger_v = try stranger.version(gpa);
    defer gpa.free(stranger_v);

    // `d` has never heard of "stranger" — naming them in `from` (the
    // lenient side) is silently ignored, unlike an unknown entry in `to`
    // (the strict side, covered above), which is a hard error.
    const batch = try d.eventsBetween(gpa, stranger_v, to);
    defer gpa.free(batch);
    var mirror: ObjectDoc = .empty;
    defer mirror.deinit(gpa);
    gpa.free(try mirror.merge(gpa, batch));
    try expectConverged(&[_]*ObjectDoc{ &d, &mirror });
}

test "openFromContent: bulk load, shared root across independent loads" {
    const gpa = t.allocator;
    var a = try ObjectDoc.openFromContent(gpa, "the same big file contents\n", "body");
    defer a.deinit(gpa);
    var b = try ObjectDoc.openFromContent(gpa, "the same big file contents\n", "body");
    defer b.deinit(gpa);
    // ONE retained event (the `map_set` that creates the text object) —
    // zero `text_ins` events regardless of content size.
    try t.expectEqual(@as(usize, 1), a.history.eventCount());
    const a_body = a.root().mapGet("body").?.objId().?;
    const a_text = try bodyText(gpa, &a, a_body);
    defer gpa.free(a_text);
    try t.expectEqualStrings("the same big file contents\n", a_text);

    // Independent loads of identical bytes share the history root: they
    // sync as replicas of one document.
    try a.setAgent(gpa, "alice");
    try b.setAgent(gpa, "bob");
    const b_body = b.root().mapGet("body").?.objId().?;
    _ = try a.textInsert(gpa, a_body, 0, "A");
    _ = try b.textInsert(gpa, b_body, b.ref(b_body).textRope().byteLen(), "B");
    try syncBoth(gpa, &a, &b);
    try expectConverged(&[_]*ObjectDoc{ &a, &b });

    // Different contents produce different bases: never confusable.
    var c = try ObjectDoc.openFromContent(gpa, "entirely different\n", "body");
    defer c.deinit(gpa);
    try c.setAgent(gpa, "carol");
    const c_body = c.root().mapGet("body").?.objId().?;
    _ = try c.textInsert(gpa, c_body, 0, "C");
    const av = try a.version(gpa);
    defer gpa.free(av);
    const batch = try c.eventsSince(gpa, av);
    defer gpa.free(batch);
    try t.expectError(error.MissingDependency, a.merge(gpa, batch));
}

test "openFromContent: invalid UTF-8 is rejected" {
    const gpa = t.allocator;
    try t.expectError(error.Corrupt, ObjectDoc.openFromContent(gpa, "\xff\xfe", "body"));
}

test "openFromContent: same content under DIFFERENT keys mints DIFFERENT roots (never silently conflated)" {
    const gpa = t.allocator;
    // Same bytes, two different field names — a real shape (e.g. loading
    // the same file as both a document's `body` and, elsewhere, its
    // `preview`). If the synthetic agent name were content-only (ignoring
    // `key`), both loads would mint the exact same `{base-<hash>, seq 0}`
    // creation-event id carrying DIFFERENT `map_set` payloads, and the two
    // docs would believe they were the same document at the same version
    // (identical `base_version` tokens, an empty diff) while silently
    // holding different keys. Folding `key` into the digest makes this a
    // loud, honest `MissingDependency` instead of silent divergence.
    var body_doc = try ObjectDoc.openFromContent(gpa, "shared bytes\n", "body");
    defer body_doc.deinit(gpa);
    var preview_doc = try ObjectDoc.openFromContent(gpa, "shared bytes\n", "preview");
    defer preview_doc.deinit(gpa);

    try t.expect(!std.mem.eql(u8, body_doc.base_version, preview_doc.base_version));
    try t.expect(body_doc.root().mapGet("body") != null);
    try t.expect(body_doc.root().mapGet("preview") == null);
    try t.expect(preview_doc.root().mapGet("preview") != null);
    try t.expect(preview_doc.root().mapGet("body") == null);

    const bv = try body_doc.version(gpa);
    defer gpa.free(bv);
    const batch = try preview_doc.eventsSince(gpa, bv);
    defer gpa.free(batch);
    try t.expectError(error.MissingDependency, body_doc.merge(gpa, batch));
}

test "openFromContent: bulk load, then edit, sync, and compact — interplay with the existing compaction machinery" {
    const gpa = t.allocator;
    var founder = try ObjectDoc.openFromContent(gpa, "line one\nline two\n", "body");
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = founder.root().mapGet("body").?.objId().?;

    var peer: ObjectDoc = .empty;
    defer peer.deinit(gpa);
    try peer.setAgent(gpa, "peer");
    try syncOne(gpa, &founder, &peer); // peer bootstraps from founder's bulk-load base
    const peer_body = peer.root().mapGet("body").?.objId().?;
    _ = try peer.textInsert(gpa, peer_body, peer.ref(peer_body).textRope().byteLen(), "line three\n");
    try syncOne(gpa, &peer, &founder);
    try expectConverged(&[_]*ObjectDoc{ &founder, &peer });

    // Compact past the bulk-loaded base's own creation point AND the new
    // edit — exercises `materializeTextBasesAt`'s progressive
    // re-compaction (seeding from the EXISTING bulk-load base, not from
    // scratch) on both replicas.
    const stable = try founder.version(gpa);
    defer gpa.free(stable);
    const before = try bodyText(gpa, &founder, body);
    defer gpa.free(before);
    try founder.compact(gpa, stable);
    try peer.compact(gpa, stable);
    const after = try bodyText(gpa, &founder, body);
    defer gpa.free(after);
    try t.expectEqualStrings(before, after);
    try expectConverged(&[_]*ObjectDoc{ &founder, &peer });

    // Serialize/open round-trips a bulk-loaded-then-edited-then-compacted
    // doc exactly like any other compacted doc.
    const bytes = try founder.serialize(gpa);
    defer gpa.free(bytes);
    var reopened = try ObjectDoc.open(gpa, bytes);
    defer reopened.deinit(gpa);
    try expectConverged(&[_]*ObjectDoc{ &founder, &reopened });

    // Sync continues past the (now doubly-advanced) compaction point.
    _ = try peer.textInsert(gpa, peer_body, 0, ">>> ");
    try syncOne(gpa, &peer, &founder);
    try expectConverged(&[_]*ObjectDoc{ &founder, &peer });
}

test "materializeAt: time travel to any known version, scoped to one text object" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "author");
    const v_empty = try d.version(gpa);
    defer gpa.free(v_empty);
    const body = (try d.mapSet(gpa, null, "body", .text)).?;

    _ = try d.textInsert(gpa, body, 0, "first draft \xf0\x9d\x84\x9e");
    const v1 = try d.version(gpa);
    defer gpa.free(v1);
    const text_v1 = try bodyText(gpa, &d, body);
    defer gpa.free(text_v1);

    _ = try d.textDelete(gpa, body, .{ .start = 0, .end = 6 });
    _ = try d.textInsert(gpa, body, 0, "FINAL");
    const v2 = try d.version(gpa);
    defer gpa.free(v2);

    var at_v1 = try d.materializeAt(gpa, body, v1);
    defer at_v1.deinit(gpa);
    const got_v1 = try at_v1.toOwnedSlice(gpa);
    defer gpa.free(got_v1);
    try t.expectEqualStrings(text_v1, got_v1);

    var at_v2 = try d.materializeAt(gpa, body, v2);
    defer at_v2.deinit(gpa);
    const got_v2 = try at_v2.toOwnedSlice(gpa);
    defer gpa.free(got_v2);
    const now = try bodyText(gpa, &d, body);
    defer gpa.free(now);
    try t.expectEqualStrings(now, got_v2);

    // `body` did not exist yet at `v_empty` (its creating `map_set` comes
    // right after) — a caller-contract violation, not a valid empty state.
    try t.expectError(error.Corrupt, d.materializeAt(gpa, body, v_empty));

    // A version we've never seen is a missing dependency.
    const unknown = "stv\x01" ++ [_]u8{ 1, 5 } ++ "ghost" ++ [_]u8{9};
    try t.expectError(error.MissingDependency, d.materializeAt(gpa, body, unknown));
}

test "materializeAt: agrees with what `compact` itself freezes into the base at the same point" {
    const gpa = t.allocator;
    // Two agents (founder creates, editor edits) — `compact`'s per-agent
    // prefix invariant rejects a single agent that both creates a text
    // object (excluded from `in_base`) and later edits it (included),
    // same landmine the "compact: text history collapses..." battery
    // above sidesteps the same way (see `compact`'s doc comment).
    var founder: ObjectDoc = .empty;
    defer founder.deinit(gpa);
    try founder.setAgent(gpa, "founder");
    const body = (try founder.mapSet(gpa, null, "body", .text)).?;

    var editor: ObjectDoc = .empty;
    defer editor.deinit(gpa);
    try editor.setAgent(gpa, "editor");
    try syncOne(gpa, &founder, &editor);
    const editor_body = editor.root().mapGet("body").?.objId().?;
    _ = try editor.textInsert(gpa, editor_body, 0, "alpha beta");
    try syncOne(gpa, &editor, &founder);

    const stable = try founder.version(gpa);
    defer gpa.free(stable);

    // `materializeAt`'s independent live-graph replay...
    var at_stable = try founder.materializeAt(gpa, body, stable);
    defer at_stable.deinit(gpa);
    const via_materialize = try at_stable.toOwnedSlice(gpa);
    defer gpa.free(via_materialize);

    // ...must agree with `compact`'s OWN materialization of the same
    // point (`materializeTextBasesAt`, a related but separate code path)
    // once `stable`'s own frontier event is folded into the base and no
    // longer decodable as a version token at all.
    _ = try editor.textInsert(gpa, editor_body, editor.ref(editor_body).textRope().byteLen(), " gamma");
    try syncOne(gpa, &editor, &founder);
    try founder.compact(gpa, stable);
    const body_creation = founder.history.idOf(founder.history.lvOf(body).?);
    const base = founder.text_bases.get(body_creation).?;
    try t.expectEqualStrings(via_materialize, base.bytes);
}

// ── Structural ops (F3, delta 6 — the move op) ──────────────────────────
// `stemma-unification.md` §3 step 5 — parent-register + fractional order
// keys, ported from `structure_sketch.zig` and re-targeted at ObjectDoc's
// real API. Hand-written cases mirror the sketch's (`structure_sketch.zig`
// ~716-1060) 1:1 where the shape carries over; the property campaign below
// is the sketch's ~400-seed campaign, trimmed (see its doc comment).

fn eqId(a: ObjectDoc.EventId, b: ObjectDoc.EventId) bool {
    return std.meta.eql(a, b);
}

/// Portable canonical dump of the structural tree (agent NAME + seq, never
/// a doc-local `ObjId`) — mirrors `structure_sketch.zig`'s
/// `Materialized.dump`, for convergence comparison across replicas.
fn structDump(gpa: std.mem.Allocator, d: *const ObjectDoc, out: *std.ArrayList(u8)) !void {
    try out.appendSlice(gpa, "ROOT\n");
    try structDumpChildren(gpa, d, out, .root, 1);
    try out.appendSlice(gpa, "TRASH\n");
    try structDumpChildren(gpa, d, out, .trash, 1);
}

fn structDumpChildren(gpa: std.mem.Allocator, d: *const ObjectDoc, out: *std.ArrayList(u8), parent: ObjectDoc.StructRef, depth: usize) !void {
    const kids = try d.structChildren(gpa, parent);
    defer gpa.free(kids);
    for (kids) |k| {
        for (0..depth) |_| try out.appendSlice(gpa, "  ");
        try out.appendSlice(gpa, d.history.agentName(k.agent));
        try out.print(gpa, "#{d}\n", .{k.seq});
        try structDumpChildren(gpa, d, out, .{ .node = k }, depth + 1);
    }
}

/// Independent oracle: walk from both roots via `structChildren`, asserting
/// every node with a resolved parent is reached exactly once (no orphans,
/// no cycles) — mirrors `structure_sketch.zig`'s `Materialized.verifyForest`,
/// does not trust any of `Walker`'s own bookkeeping.
fn verifyStructForest(gpa: std.mem.Allocator, d: *const ObjectDoc, total_nodes: usize) !void {
    var visited: std.AutoHashMapUnmanaged(ObjectDoc.EventId, void) = .empty;
    defer visited.deinit(gpa);
    try structWalkCount(gpa, d, .root, &visited);
    try structWalkCount(gpa, d, .trash, &visited);
    try t.expectEqual(total_nodes, visited.count());
}

fn structWalkCount(gpa: std.mem.Allocator, d: *const ObjectDoc, parent: ObjectDoc.StructRef, visited: *std.AutoHashMapUnmanaged(ObjectDoc.EventId, void)) !void {
    const kids = try d.structChildren(gpa, parent);
    defer gpa.free(kids);
    for (kids) |k| {
        const gop = try visited.getOrPut(gpa, k);
        try t.expect(!gop.found_existing); // CycleDetected / double-parented
        try structWalkCount(gpa, d, .{ .node = k }, visited);
    }
}

test "structCreate + structMove: children read back sorted, one register write per move" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const mid = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(mid);
    const a = try d.structCreate(gpa, .root, mid);
    const before_count = d.history.eventCount();

    const k2 = try ObjectDoc.orderKeyBetween(gpa, mid, null);
    defer gpa.free(k2);
    const b = try d.structCreate(gpa, .root, k2);

    const roots1 = try d.structChildren(gpa, .root);
    defer gpa.free(roots1);
    try t.expectEqual(@as(usize, 2), roots1.len);
    try t.expect(eqId(roots1[0], a));
    try t.expect(eqId(roots1[1], b));

    const k3 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k3);
    try d.structMove(gpa, b, .{ .node = a }, k3);
    try t.expectEqual(before_count + 2, d.history.eventCount());

    const roots2 = try d.structChildren(gpa, .root);
    defer gpa.free(roots2);
    try t.expectEqual(@as(usize, 1), roots2.len);
    const a_kids = try d.structChildren(gpa, .{ .node = a });
    defer gpa.free(a_kids);
    try t.expectEqual(@as(usize, 1), a_kids.len);
    try t.expect(eqId(a_kids[0], b));
    try verifyStructForest(gpa, &d, 2);
}

test "a structural node doubles as an ordinary map object" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const key = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(key);
    const node = try d.structCreate(gpa, .root, key);
    _ = try d.mapSet(gpa, node, "title", .{ .str = "hello" });
    const ref = d.ref(node);
    try t.expectEqualStrings("hello", ref.mapGet("title").?.asStr());
}

test "structCreate/structMove refuse an order key longer than the wire cap, in every build mode" {
    // `assert` would compile out under ReleaseFast/ReleaseSmall (silently
    // reopening the un-round-trippable hole `max_order_key_len`'s doc
    // comment closes) and would crash THIS suite outright even in Debug
    // — the very reason this has to be a real error, not an assert, and
    // the very reason it's untestable any other way: a test asserting an
    // `assert` fires is a test that kills the test binary.
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const too_long = try gpa.alloc(u8, 4097);
    defer gpa.free(too_long);
    @memset(too_long, 'x');
    try t.expectError(error.OrderKeyTooLong, d.structCreate(gpa, .root, too_long));
    try t.expectEqual(@as(usize, 0), d.history.eventCount()); // refused before any mutation

    const ok_key = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(ok_key);
    const node = try d.structCreate(gpa, .root, ok_key);
    try t.expectError(error.OrderKeyTooLong, d.structMove(gpa, node, .root, too_long));
    // The refused move's own event IS recorded (only the STRUCT_CREATE
    // above was refused pre-append) — `structMove`'s length check runs
    // before `addLocal` too, so no new event exists for this rejection.
    const events_before = d.history.eventCount();
    try t.expectError(error.OrderKeyTooLong, d.structMove(gpa, node, .root, too_long));
    try t.expectEqual(events_before, d.history.eventCount());
}

test "concurrent structMove of the same node: both survive as conflicts, deterministic winner, both replicas converge" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    const shared = try alice.structCreate(gpa, .root, k0);
    const k1 = try ObjectDoc.orderKeyBetween(gpa, k0, null);
    defer gpa.free(k1);
    const parent_a = try alice.structCreate(gpa, .root, k1);
    const k2 = try ObjectDoc.orderKeyBetween(gpa, k1, null);
    defer gpa.free(k2);
    const parent_b = try alice.structCreate(gpa, .root, k2);
    try syncOne(gpa, &alice, &bob);

    const ka = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(ka);
    try alice.structMove(gpa, shared, .{ .node = parent_a }, ka);
    // `shared`/`parent_b` are ALICE-numbered `ObjId`s (`EventId.agent` is
    // replica-local — causal.zig) — translate through bob's own agent
    // table before using them in a call against `bob`, exactly like the
    // existing map-conflict tests above do for cross-replica `ObjId`s.
    const bob_shared: ObjectDoc.ObjId = .{ .agent = bob.history.findAgent("alice").?, .seq = shared.seq };
    const bob_parent_b: ObjectDoc.ObjId = .{ .agent = bob.history.findAgent("alice").?, .seq = parent_b.seq };
    const kb = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(kb);
    try bob.structMove(gpa, bob_shared, .{ .node = bob_parent_b }, kb);

    try syncBoth(gpa, &alice, &bob);

    var dump_a: std.ArrayList(u8) = .empty;
    defer dump_a.deinit(gpa);
    try structDump(gpa, &alice, &dump_a);
    var dump_b: std.ArrayList(u8) = .empty;
    defer dump_b.deinit(gpa);
    try structDump(gpa, &bob, &dump_b);
    try t.expectEqualStrings(dump_a.items, dump_b.items);

    // Honest MV: 2 concurrent writes to `shared`'s register both survive.
    try t.expectEqual(@as(usize, 2), alice.structConflictCount(shared));
    try t.expectEqual(@as(usize, 2), bob.structConflictCount(bob_shared));
    try t.expect(!alice.structCycleBroken(shared));

    const winner = alice.structParent(shared).?;
    try t.expect(winner == .node);
    try t.expect(eqId(winner.node, parent_a) or eqId(winner.node, parent_b));
    try verifyStructForest(gpa, &alice, 3);
}

test "concurrent moves that individually are acyclic but jointly cycle: deterministically resolved, stays acyclic, both replicas agree" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    const na = try alice.structCreate(gpa, .root, k0);
    const k1 = try ObjectDoc.orderKeyBetween(gpa, k0, null);
    defer gpa.free(k1);
    const nb = try alice.structCreate(gpa, .root, k1);
    try syncOne(gpa, &alice, &bob);

    const ka = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(ka);
    try alice.structMove(gpa, na, .{ .node = nb }, ka);
    // Translate through bob's own agent table — see the note in the
    // "concurrent structMove of the same node" test above.
    const bob_na: ObjectDoc.ObjId = .{ .agent = bob.history.findAgent("alice").?, .seq = na.seq };
    const bob_nb: ObjectDoc.ObjId = .{ .agent = bob.history.findAgent("alice").?, .seq = nb.seq };
    const kb = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(kb);
    try bob.structMove(gpa, bob_nb, .{ .node = bob_na }, kb);

    try syncBoth(gpa, &alice, &bob);

    try verifyStructForest(gpa, &alice, 2);
    try verifyStructForest(gpa, &bob, 2);

    var dump_a: std.ArrayList(u8) = .empty;
    defer dump_a.deinit(gpa);
    try structDump(gpa, &alice, &dump_a);
    var dump_b: std.ArrayList(u8) = .empty;
    defer dump_b.deinit(gpa);
    try structDump(gpa, &bob, &dump_b);
    try t.expectEqualStrings(dump_a.items, dump_b.items);
}

// Pins the FINDING documented on `ObjectDoc.structParent`: a rejected write
// can strand an earlier, already-superseded write as the effective parent,
// outside the reported conflict set. Single replica, fully sequential (no
// MV conflict at all — this is about cycle-rejection, not concurrency):
// create N; create P; move P under N (accepted); move N under P (would
// make N its own ancestor via P — rejected). N's effective parent falls
// all the way back to its own `structCreate` (`.root`), but `structCreate`
// was already causally superseded in the conflict-set bookkeeping by the
// (rejected) move — see `structure_sketch.zig`'s pinned counterexample
// test (~875-915) for the exact mechanism this mirrors.
test "a rejected write can leave an earlier, already-superseded write as the effective parent — outside the reported conflict set" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const kn = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(kn);
    const n = try d.structCreate(gpa, .root, kn);
    const kp = try ObjectDoc.orderKeyBetween(gpa, kn, null);
    defer gpa.free(kp);
    const p = try d.structCreate(gpa, .root, kp);

    const k1 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k1);
    try d.structMove(gpa, p, .{ .node = n }, k1); // P under N: fine, no cycle yet

    const k2 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k2);
    try d.structMove(gpa, n, .{ .node = p }, k2); // N under P: would cycle — local apply refuses

    try verifyStructForest(gpa, &d, 2);

    // The EFFECTIVE PARENT: N's own `create`, all the way back — not the
    // rejected move.
    const winner = d.structParent(n).?;
    try t.expect(winner == .root);
    // The CONVERGENCE PROPERTY this test actually pins: the AUTHORING
    // replica's own accessors must already agree with what a canonical
    // (Lamport-order) resolution of this exact history reports — not
    // some stale pre-refusal snapshot. The refused move's causal parents
    // are the frontier at the time it was written, which dominates every
    // prior write to N's register (including N's own `create`) — so it
    // supersedes them in the antichain regardless of being rejected,
    // collapsing `conflict_live` to size 1 with the WINNER outside it.
    // `structMove`'s refusal branch must update this metadata even
    // though the effective parent itself doesn't change — see that
    // function's doc comment for the invariant.
    try t.expectEqual(@as(usize, 1), d.structConflictCount(n));
    try t.expect(d.structCycleBroken(n));

    // Now replay the SAME history through a fresh replica via the wire —
    // this exercises `objects_state.Walker`'s GLOBAL Lamport-canonical
    // resolution (the standalone `resolveStructs` pass, not `structMove`'s
    // local fast path above) — and must report EXACTLY the same thing.
    // Two replicas holding identical history must never disagree on a
    // documented accessor, even when the disagreement would only ever be
    // about "conflict" bookkeeping metadata rather than the (correct,
    // convergent) effective parent itself.
    var other: ObjectDoc = .empty;
    defer other.deinit(gpa);
    try other.setAgent(gpa, "other");
    try syncOne(gpa, &d, &other);

    try verifyStructForest(gpa, &other, 2);
    const other_n: ObjectDoc.ObjId = .{ .agent = other.history.findAgent("solo").?, .seq = n.seq };
    const other_winner = other.structParent(other_n).?;
    try t.expect(other_winner == .root);
    try t.expectEqual(@as(usize, 1), other.structConflictCount(other_n));
    try t.expect(other.structCycleBroken(other_n));

    // The two replicas AGREE — the actual property, not just each one's
    // own internal self-consistency.
    try t.expectEqual(d.structConflictCount(n), other.structConflictCount(other_n));
    try t.expectEqual(d.structCycleBroken(n), other.structCycleBroken(other_n));
}

test "structDelete hides a subtree without destroying it; undelete restores it intact" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    const parent = try d.structCreate(gpa, .root, k0);
    const k1 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k1);
    const child = try d.structCreate(gpa, .{ .node = parent }, k1);

    try d.structDelete(gpa, parent);

    const roots1 = try d.structChildren(gpa, .root);
    defer gpa.free(roots1);
    try t.expectEqual(@as(usize, 0), roots1.len);
    const trash_kids = try d.structChildren(gpa, .trash);
    defer gpa.free(trash_kids);
    try t.expectEqual(@as(usize, 1), trash_kids.len);
    try t.expect(eqId(trash_kids[0], parent));
    const under_trashed = try d.structChildren(gpa, .{ .node = parent });
    defer gpa.free(under_trashed);
    try t.expectEqual(@as(usize, 1), under_trashed.len);
    try t.expect(eqId(under_trashed[0], child));
    try verifyStructForest(gpa, &d, 2);

    const k2 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k2);
    try d.structMove(gpa, parent, .root, k2);

    const roots2 = try d.structChildren(gpa, .root);
    defer gpa.free(roots2);
    try t.expectEqual(@as(usize, 1), roots2.len);
    const restored_kids = try d.structChildren(gpa, .{ .node = parent });
    defer gpa.free(restored_kids);
    try t.expectEqual(@as(usize, 1), restored_kids.len);
    try t.expect(eqId(restored_kids[0], child));
    try verifyStructForest(gpa, &d, 2);
}

test "untouched siblings keep relative order across merges" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    var prev: ?[]u8 = null;
    var untouched: [3]ObjectDoc.ObjId = undefined;
    for (0..3) |i| {
        const k = try ObjectDoc.orderKeyBetween(gpa, prev, null);
        defer gpa.free(k);
        untouched[i] = try alice.structCreate(gpa, .root, k);
        if (prev) |p| gpa.free(p);
        prev = try gpa.dupe(u8, k);
    }
    if (prev) |p| gpa.free(p);
    try syncOne(gpa, &alice, &bob);

    const bob_roots = try bob.structChildren(gpa, .root);
    defer gpa.free(bob_roots);
    const bk = try ObjectDoc.orderKeyBetween(gpa, bob.structOrderKey(bob_roots[0]), bob.structOrderKey(bob_roots[1]));
    defer gpa.free(bk);
    _ = try bob.structCreate(gpa, .root, bk);

    try syncBoth(gpa, &alice, &bob);

    const final_roots = try alice.structChildren(gpa, .root);
    defer gpa.free(final_roots);
    try t.expectEqual(@as(usize, 4), final_roots.len);

    var positions: [3]usize = undefined;
    for (untouched, 0..) |u, ui| {
        for (final_roots, 0..) |r, ri| {
            if (eqId(r, u)) positions[ui] = ri;
        }
    }
    try t.expect(positions[0] < positions[1]);
    try t.expect(positions[1] < positions[2]);
}

test "wire round-trip: structural frames survive serialize/open, structure-free docs are unaffected" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    const a = try d.structCreate(gpa, .root, k0);
    _ = try d.mapSet(gpa, a, "n", .{ .int = 1 });
    const k1 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k1);
    const b = try d.structCreate(gpa, .{ .node = a }, k1);
    _ = b;
    try d.structDelete(gpa, a);

    const bytes = try d.serialize(gpa);
    defer gpa.free(bytes);
    var re = try ObjectDoc.open(gpa, bytes);
    defer re.deinit(gpa);

    var d1: std.ArrayList(u8) = .empty;
    defer d1.deinit(gpa);
    try structDump(gpa, &d, &d1);
    var d2: std.ArrayList(u8) = .empty;
    defer d2.deinit(gpa);
    try structDump(gpa, &re, &d2);
    try t.expectEqualStrings(d1.items, d2.items);
    try verifyStructForest(gpa, &re, 2);
}

test "wire: a doc with no structural ops carries no struct_create/struct_move tag bytes" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");
    _ = try d.mapSet(gpa, null, "a", .{ .int = 1 });
    const list = (try d.mapSet(gpa, null, "l", .list)).?;
    _ = try d.listInsert(gpa, list, 0, .{ .int = 2 });

    const bytes = try d.serialize(gpa);
    defer gpa.free(bytes);
    // Every op-tag byte in a struct-op-free stream is <= 5 (the pre-delta-6
    // tag range) — a hand-decoded scan confirms the encoder never touches
    // tags 6/7 unless a structural op is actually present, i.e. wire bytes
    // for a structure-free doc are exactly what pre-delta-6 code emitted.
    var re = try ObjectDoc.open(gpa, bytes);
    defer re.deinit(gpa);
    const got = try re.toJson(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings(
        \\{"a":1,"l":[2]}
    , got);
}

test "old wire bytes (tags 0-5 only, hand-built) still decode under the delta-6 decoder" {
    const gpa = t.allocator;
    // Hand-built v1 batch: one agent "solo", one event — map_set root.a=7.
    // Exactly the pre-delta-6 wire shape (OpTag values 0-5 only,
    // `object_magic_v1`) — proves the extended decoder (tag bound now 7,
    // not 5) still accepts every byte a pre-delta-6 encoder ever produced.
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    try bytes.appendSlice(gpa, "stj\x01");
    try bytes.append(gpa, 1); // agent_count
    try bytes.append(gpa, 4); // name len
    try bytes.appendSlice(gpa, "solo");
    try bytes.append(gpa, 1); // event_count
    try bytes.append(gpa, 0); // agent_idx
    try bytes.append(gpa, 0); // seq
    try bytes.append(gpa, 0); // parent_count
    try bytes.append(gpa, 0); // op tag: map_set
    try bytes.append(gpa, 0); // has_obj = 0 (root)
    try bytes.append(gpa, 1); // key len
    try bytes.appendSlice(gpa, "a");
    try bytes.append(gpa, 3); // ValTag.int
    try bytes.append(gpa, 14); // zigzag(7) = 14

    var d = try ObjectDoc.open(gpa, bytes.items);
    defer d.deinit(gpa);
    try t.expectEqual(@as(i64, 7), d.root().mapGet("a").?.asInt());
}

test "compact: refuses when a structural op is in the causal past" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");
    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    _ = try d.structCreate(gpa, .root, k0);

    const stable = try d.version(gpa);
    defer gpa.free(stable);
    try t.expectError(error.NotCompactable, d.compact(gpa, stable));
}

test "compact still works on a doc whose structural ops are all AFTER the stable point" {
    const gpa = t.allocator;
    var d: ObjectDoc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");
    // A plain scalar map write (never `in_base`-eligible either way — see
    // `compact`'s doc comment) rather than a text object, to stay clear of
    // the SEPARATE, pre-existing "same agent creates then edits an object
    // straddling the stable point" prefix limitation that doc comment
    // names (unrelated to structural ops).
    _ = try d.mapSet(gpa, null, "title", .{ .str = "doc" });

    const stable = try d.version(gpa);
    defer gpa.free(stable);
    try d.compact(gpa, stable);

    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    const a = try d.structCreate(gpa, .root, k0);
    const kids = try d.structChildren(gpa, .root);
    defer gpa.free(kids);
    try t.expectEqual(@as(usize, 1), kids.len);
    try t.expect(eqId(kids[0], a));
}

test "moves + text edits + map writes interleaved: converges, structure and content both correct" {
    const gpa = t.allocator;
    var alice: ObjectDoc = .empty;
    defer alice.deinit(gpa);
    var bob: ObjectDoc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const k0 = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(k0);
    const a = try alice.structCreate(gpa, .root, k0);
    _ = try alice.mapSet(gpa, a, "title", .{ .str = "note" });
    const k1 = try ObjectDoc.orderKeyBetween(gpa, k0, null);
    defer gpa.free(k1);
    const b = try alice.structCreate(gpa, .root, k1);
    try syncOne(gpa, &alice, &bob);

    // Concurrent: alice edits a map key on `a` AND moves `b` under `a`;
    // bob independently edits text and also moves `b`'s sibling ordering.
    _ = try alice.mapSet(gpa, a, "title", .{ .str = "note!" });
    const ka = try ObjectDoc.orderKeyBetween(gpa, null, null);
    defer gpa.free(ka);
    try alice.structMove(gpa, b, .{ .node = a }, ka);

    const bob_a: ObjectDoc.ObjId = .{ .agent = bob.history.findAgent("alice").?, .seq = a.seq };
    _ = try bob.mapSet(gpa, bob_a, "note_body", .text);
    const body_obj = bob.ref(bob_a).mapGet("note_body").?.objId().?;
    _ = try bob.textInsert(gpa, body_obj, 0, "hi");

    try syncBoth(gpa, &alice, &bob);

    try t.expectEqualStrings("note!", alice.ref(a).mapGet("title").?.asStr());
    try t.expectEqualStrings("note!", bob.ref(bob_a).mapGet("title").?.asStr());
    var da: std.ArrayList(u8) = .empty;
    defer da.deinit(gpa);
    try structDump(gpa, &alice, &da);
    var db: std.ArrayList(u8) = .empty;
    defer db.deinit(gpa);
    try structDump(gpa, &bob, &db);
    try t.expectEqualStrings(da.items, db.items);
    try verifyStructForest(gpa, &alice, 2);

    const alice_json = try alice.toJson(gpa);
    defer gpa.free(alice_json);
    const bob_json = try bob.toJson(gpa);
    defer gpa.free(bob_json);
    try t.expectEqualStrings(alice_json, bob_json);
}

// ── Property campaign (F3, delta 6) ─────────────────────────────────────
// Ported from `structure_sketch.zig`'s ~400-seed / ~30-op / up-to-4-replica
// campaign (~1095-1192), re-targeted at ObjectDoc's real API (structural
// ops interleaved with map/list/text edits, going through the SAME
// `merge`/`Walker` path production code uses — not a standalone sketch
// `Doc`). TRIMMED to 80 seeds / 20 ops / up to 3 replicas: `ObjectDoc`'s
// replay is materially heavier per call than the sketch's (map/list/text
// state plus two structural canonical-order passes, all recomputed on
// every `merge`, per `objects_state.Walker`'s doc comment) — 400 seeds at
// the sketch's op count made this suite noticeably slower without
// exercising new code paths past roughly the first several dozen seeds
// (each schedule is i.i.d.; the interesting shapes — MV conflicts,
// cross-node cycles, trash/resurrection — all recur well within 80).
// Asserts, every seed: convergence (byte-identical structural dumps AND
// canonical JSON) and acyclicity + full reachability
// (`verifyStructForest`, independent of `Walker`'s own bookkeeping).
test "property: random structural + map/list/text schedules across 2-3 replicas converge, stay acyclic, stay reachable" {
    const gpa = t.allocator;
    const schedules = 80;
    var seed: u64 = 0;
    while (seed < schedules) : (seed += 1) {
        runStructSchedule(gpa, seed) catch |err| {
            std.debug.print("seed={d} failed\n", .{seed});
            return err;
        };
    }
}

const NodeHandle = struct { name: []const u8, seq: u64 };

fn runStructSchedule(gpa: std.mem.Allocator, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const replica_count = 2 + random.uintLessThan(usize, 2); // 2..3

    var docs: [3]ObjectDoc = .{ .empty, .empty, .empty };
    defer for (0..replica_count) |i| docs[i].deinit(gpa);
    const names = [_][]const u8{ "r0", "r1", "r2" };
    for (0..replica_count) |i| try docs[i].setAgent(gpa, names[i]);

    var handles: std.ArrayList(NodeHandle) = .empty;
    defer handles.deinit(gpa);
    var total_nodes: usize = 0;

    const op_count = 20;
    for (0..op_count) |_| {
        const ri = random.uintLessThan(usize, replica_count);
        const d = &docs[ri];
        const choice = random.uintLessThan(u8, 6);
        switch (choice) {
            0, 1 => { // structCreate (weighted higher — material to move)
                const parent = (try pickStructTarget(gpa, d, random, handles.items, true)).?;
                const kids = try d.structChildren(gpa, parent);
                defer gpa.free(kids);
                const digits = try randomGapKey(gpa, d, kids, random);
                defer gpa.free(digits);
                const id = try d.structCreate(gpa, parent, digits);
                try handles.append(gpa, .{ .name = d.history.agentName(id.agent), .seq = id.seq });
                total_nodes += 1;
            },
            2 => { // structMove / reorder
                const target = try pickStructTarget(gpa, d, random, handles.items, false) orelse continue;
                if (target != .node) continue;
                const parent = (try pickStructTarget(gpa, d, random, handles.items, true)).?;
                const kids = try d.structChildren(gpa, parent);
                defer gpa.free(kids);
                const digits = try randomGapKey(gpa, d, kids, random);
                defer gpa.free(digits);
                try d.structMove(gpa, target.node, parent, digits);
            },
            3 => { // structDelete
                const target = try pickStructTarget(gpa, d, random, handles.items, false) orelse continue;
                if (target != .node) continue;
                try d.structDelete(gpa, target.node);
            },
            4 => { // an ordinary map write on a random known node — cross-boundary interaction
                const target = try pickStructTarget(gpa, d, random, handles.items, false) orelse continue;
                if (target != .node) continue;
                _ = d.mapSet(gpa, target.node, "tag", .{ .int = @intCast(seed) }) catch |e| switch (e) {
                    error.OutOfMemory => return e,
                };
            },
            5 => { // sync a random pair, both directions
                if (replica_count < 2) continue;
                var i = random.uintLessThan(usize, replica_count);
                var j = random.uintLessThan(usize, replica_count);
                if (i == j) j = (j + 1) % replica_count;
                try syncOne(gpa, &docs[j], &docs[i]);
                i = random.uintLessThan(usize, replica_count);
                j = random.uintLessThan(usize, replica_count);
                if (i == j) j = (j + 1) % replica_count;
                try syncOne(gpa, &docs[j], &docs[i]);
            },
            else => unreachable,
        }
    }

    for (0..2) |_| {
        for (0..replica_count) |i| {
            for (0..replica_count) |j| {
                if (i != j) try syncOne(gpa, &docs[j], &docs[i]);
            }
        }
    }

    var dumps: [3]std.ArrayList(u8) = .{ .empty, .empty, .empty };
    defer for (0..replica_count) |i| dumps[i].deinit(gpa);
    var jsons: [3][]u8 = undefined;
    defer for (0..replica_count) |i| gpa.free(jsons[i]);
    for (0..replica_count) |i| {
        try verifyStructForest(gpa, &docs[i], total_nodes);
        try structDump(gpa, &docs[i], &dumps[i]);
        jsons[i] = try docs[i].toJson(gpa);
    }
    for (1..replica_count) |i| {
        try t.expectEqualStrings(dumps[0].items, dumps[i].items);
        try t.expectEqualStrings(jsons[0], jsons[i]);
    }
}

fn pickStructTarget(
    gpa: std.mem.Allocator,
    d: *const ObjectDoc,
    random: std.Random,
    handles: []const NodeHandle,
    allow_special: bool,
) !?ObjectDoc.StructRef {
    var known: std.ArrayList(ObjectDoc.ObjId) = .empty;
    defer known.deinit(gpa);
    for (handles) |h| {
        if (d.history.findAgent(h.name)) |aid| {
            const id: ObjectDoc.EventId = .{ .agent = aid, .seq = h.seq };
            if (d.history.isKnown(id)) try known.append(gpa, id);
        }
    }
    if (allow_special) {
        const pool = known.items.len + 2;
        const pick = random.uintLessThan(usize, pool);
        if (pick == 0) return .root;
        if (pick == 1) return .trash;
        return .{ .node = known.items[pick - 2] };
    }
    if (known.items.len == 0) return null;
    return .{ .node = known.items[random.uintLessThan(usize, known.items.len)] };
}

fn randomGapKey(gpa: std.mem.Allocator, d: *const ObjectDoc, kids: []const ObjectDoc.ObjId, random: std.Random) ![]u8 {
    if (kids.len == 0) return ObjectDoc.orderKeyBetween(gpa, null, null);
    const gap = random.uintLessThan(usize, kids.len + 1);
    var a: ?[]const u8 = if (gap == 0) null else d.structOrderKey(kids[gap - 1]);
    var b: ?[]const u8 = if (gap == kids.len) null else d.structOrderKey(kids[gap]);
    if (a != null and b != null and std.mem.eql(u8, a.?, b.?)) {
        var lo = gap;
        while (lo > 0 and std.mem.eql(u8, d.structOrderKey(kids[lo - 1]).?, a.?)) lo -= 1;
        var hi = gap;
        while (hi < kids.len and std.mem.eql(u8, d.structOrderKey(kids[hi]).?, b.?)) hi += 1;
        a = if (lo == 0) null else d.structOrderKey(kids[lo - 1]);
        b = if (hi == kids.len) null else d.structOrderKey(kids[hi]);
    }
    return ObjectDoc.orderKeyBetween(gpa, a, b);
}
