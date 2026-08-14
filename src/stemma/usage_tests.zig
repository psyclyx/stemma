//! Executable documentation: the editor features stemma deliberately does
//! NOT ship — undo/redo, multi-cursor, search-and-replace — built from the
//! primitives it does. If composing one of these ever gets awkward, that is
//! an API bug; these tests are the early-warning system.

const std = @import("std");
const t = std.testing;

const rope_mod = @import("rope.zig");
const geometry = @import("geometry.zig");
const AnchorSet = @import("AnchorSet.zig");

const Range = geometry.Range;
const Anchor = geometry.Anchor;
const Edit = geometry.Edit;

// Small chunks so these exercise real tree structure.
const Rope = rope_mod.RopeWith(.{ .chunk_capacity = 16, .thread_safe = false });

test "undo/redo: a snapshot stack over O(1) snapshots" {
    const gpa = t.allocator;

    // The entire undo engine an editor needs from the buffer's side:
    // push snapshot() before each undo group, restore by swapping ropes.
    const History = struct {
        undo: std.ArrayList(Rope) = .empty,
        redo: std.ArrayList(Rope) = .empty,

        fn deinit(self: *@This(), a: std.mem.Allocator) void {
            for (self.undo.items) |*r| r.deinit(a);
            for (self.redo.items) |*r| r.deinit(a);
            self.undo.deinit(a);
            self.redo.deinit(a);
        }

        fn checkpoint(self: *@This(), a: std.mem.Allocator, doc: *const Rope) !void {
            try self.undo.append(a, doc.snapshot());
            for (self.redo.items) |*r| r.deinit(a);
            self.redo.clearRetainingCapacity();
        }

        fn undoOne(self: *@This(), a: std.mem.Allocator, doc: *Rope) !bool {
            const prev = self.undo.pop() orelse return false;
            try self.redo.append(a, doc.*);
            doc.* = prev;
            return true;
        }

        fn redoOne(self: *@This(), a: std.mem.Allocator, doc: *Rope) !bool {
            const next = self.redo.pop() orelse return false;
            try self.undo.append(a, doc.*);
            doc.* = next;
            return true;
        }
    };

    var doc = try Rope.fromSlice(gpa, "the original text");
    defer doc.deinit(gpa);
    var hist: History = .{};
    defer hist.deinit(gpa);

    try hist.checkpoint(gpa, &doc);
    _ = try doc.replace(gpa, .{ .start = 4, .end = 12 }, "edited");
    try hist.checkpoint(gpa, &doc);
    _ = try doc.insert(gpa, doc.byteLen(), ", twice");

    const s2 = try doc.toOwnedSlice(gpa);
    defer gpa.free(s2);
    try t.expectEqualStrings("the edited text, twice", s2);

    try t.expect(try hist.undoOne(gpa, &doc));
    const s1 = try doc.toOwnedSlice(gpa);
    defer gpa.free(s1);
    try t.expectEqualStrings("the edited text", s1);

    try t.expect(try hist.undoOne(gpa, &doc));
    const s0 = try doc.toOwnedSlice(gpa);
    defer gpa.free(s0);
    try t.expectEqualStrings("the original text", s0);

    try t.expect(try hist.redoOne(gpa, &doc));
    try t.expect(try hist.redoOne(gpa, &doc));
    const s3 = try doc.toOwnedSlice(gpa);
    defer gpa.free(s3);
    try t.expectEqualStrings("the edited text, twice", s3);
    try t.expect(!try hist.redoOne(gpa, &doc));
}

test "multi-cursor: carets in an AnchorSet, simultaneous typing" {
    const gpa = t.allocator;
    var doc = try Rope.fromSlice(gpa, "aaa bbb ccc ddd");
    defer doc.deinit(gpa);
    var carets: AnchorSet = .empty;
    defer carets.deinit(gpa);

    // A caret after each word; right bias so they ride inserted text.
    for ([_]usize{ 3, 7, 11, 15 }) |off| {
        _ = try carets.add(gpa, .{ .offset = off, .bias = .right });
    }

    // "Type" at every caret: collect positions (already sorted), apply
    // right-to-left so earlier offsets stay valid, shift the whole set
    // through each edit.
    var positions: std.ArrayList(usize) = .empty;
    defer positions.deinit(gpa);
    var it = carets.inRange(.{ .start = 0, .end = doc.byteLen() + 1 });
    while (it.next()) |e| try positions.append(gpa, e.anchor.offset);

    var i = positions.items.len;
    while (i > 0) {
        i -= 1;
        const edit = try doc.insert(gpa, positions.items[i], "!");
        carets.shift(edit);
    }

    const out = try doc.toOwnedSlice(gpa);
    defer gpa.free(out);
    try t.expectEqualStrings("aaa! bbb! ccc! ddd!", out);

    // Every caret sits after its inserted bang.
    var it2 = carets.inRange(.{ .start = 0, .end = doc.byteLen() + 1 });
    for ([_]usize{ 4, 9, 14, 19 }) |expected| {
        try t.expectEqual(expected, it2.next().?.anchor.offset);
    }
}

test "search and replace-all across chunk boundaries" {
    const gpa = t.allocator;
    var doc = try Rope.fromSlice(gpa, "one needle, two needle, red needle, blue needle");
    defer doc.deinit(gpa);

    // Collect matches first (edits invalidate iterators), then apply
    // right-to-left so match offsets stay valid without shifting.
    var hits: std.ArrayList(usize) = .empty;
    defer hits.deinit(gpa);
    var it = doc.findIterator(.{ .start = 0, .end = doc.byteLen() }, "needle");
    while (it.next()) |off| try hits.append(gpa, off);
    try t.expectEqual(@as(usize, 4), hits.items.len);

    var i = hits.items.len;
    while (i > 0) {
        i -= 1;
        _ = try doc.replace(gpa, .{ .start = hits.items[i], .end = hits.items[i] + 6 }, "thread");
    }

    const out = try doc.toOwnedSlice(gpa);
    defer gpa.free(out);
    try t.expectEqualStrings("one thread, two thread, red thread, blue thread", out);
}

test "viewport render loop: line iterator + LSP positions" {
    const gpa = t.allocator;
    var doc = try Rope.fromSlice(gpa, "fn main() void {\n    print(\"héllo\");\n    // 𝄞 done\n}\n");
    defer doc.deinit(gpa);

    // Render lines 1..3 the way a viewport would.
    var lines = doc.lineIterator(1);
    const l1 = lines.next().?;
    var buf: [64]u8 = undefined;
    doc.copyRange(buf[0..l1.len()], l1);
    try t.expectEqualStrings("    print(\"héllo\");", buf[0..l1.len()]);

    // An LSP diagnostic at the 𝄞 (astral scalar → 2 UTF-16 units).
    const music = doc.find(.{ .start = 0, .end = doc.byteLen() }, "𝄞").?;
    const pos = doc.offsetToPointUtf16(music);
    try t.expectEqual(@as(usize, 2), pos.row);
    try t.expectEqual(@as(usize, 7), pos.col);
    try t.expectEqual(music, doc.pointUtf16ToOffset(pos));
}
