//! Behavioral tests for the rope: randomized edit sequences checked against a
//! naive ArrayList/std.unicode oracle, snapshot-immutability under continued
//! edits, structural ops, borrowed backing, and streaming construction.
//! Everything runs under std.testing.allocator, so leaks and double-frees in
//! the refcount machinery fail the suite.

const std = @import("std");
const t = std.testing;

const rope_mod = @import("rope.zig");
const geometry = @import("geometry.zig");
const Range = geometry.Range;
const Point = geometry.Point;

const Rope = rope_mod.Rope;
/// Tiny nodes force deep trees, constant splitting/merging, and the
/// non-atomic refcount path — the structural engine's stress instantiation.
const TinyRope = rope_mod.RopeWith(.{
    .chunk_capacity = 8,
    .borrowed_capacity = 16,
    .branch_factor = 3,
    .thread_safe = false,
});

// ── Reference oracle ────────────────────────────────────────────────────

const pool = [_][]const u8{
    "a",                      "bc", "\n",
    "héllo",
    "wörld\n",
    "𝄞music",
    "日本語",
    "x\ny\nz\n",
    "€",
    " the quick brown fox\n",
};

fn refPoint(bytes: []const u8, off: usize) Point {
    var row: u32 = 0;
    var line_start: usize = 0;
    for (bytes[0..off], 0..) |b, i| {
        if (b == '\n') {
            row += 1;
            line_start = i + 1;
        }
    }
    return .{ .row = row, .col = @intCast(off - line_start) };
}

/// All scalar-boundary offsets of `bytes`, including 0 and len.
fn refBoundaries(gpa: std.mem.Allocator, bytes: []const u8) !std.ArrayList(usize) {
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    for (bytes, 0..) |b, i| {
        if (b & 0xC0 != 0x80) try out.append(gpa, i);
    }
    try out.append(gpa, bytes.len);
    return out;
}

fn checkAgainstReference(r: anytype, ref: []const u8) !void {
    const gpa = t.allocator;

    // Content.
    try t.expectEqual(ref.len, r.byteLen());
    const got = try r.toOwnedSlice(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings(ref, got);

    // Metrics.
    try t.expectEqual(try std.unicode.utf8CountCodepoints(ref), r.scalarLen());
    try t.expectEqual(try std.unicode.calcUtf16LeLen(ref), r.utf16Len());
    try t.expectEqual(std.mem.count(u8, ref, "\n") + 1, r.lineCount());

    // Conversions at a spread of boundaries.
    var bounds = try refBoundaries(gpa, ref);
    defer bounds.deinit(gpa);
    const step = @max(1, bounds.items.len / 7);
    var i: usize = 0;
    while (i < bounds.items.len) : (i += step) {
        const off = bounds.items[i];
        const p = r.offsetToPoint(off);
        try t.expectEqual(refPoint(ref, off), p);
        try t.expectEqual(off, r.pointToOffset(p));
        try t.expectEqual(try std.unicode.utf8CountCodepoints(ref[0..off]), r.offsetToScalar(off));
        try t.expectEqual(off, r.scalarToOffset(r.offsetToScalar(off)));
        try t.expectEqual(try std.unicode.calcUtf16LeLen(ref[0..off]), r.offsetToUtf16(off));
        try t.expectEqual(off, r.utf16ToOffset(r.offsetToUtf16(off)));
    }

    // Line ranges.
    {
        var row: usize = 0;
        var it = std.mem.splitScalar(u8, ref, '\n');
        while (it.next()) |line| : (row += 1) {
            const lr = r.lineRange(row);
            try t.expectEqualStrings(line, got[lr.start..lr.end]);
        }
        try t.expectEqual(row, r.lineCount());
    }

    // Chunk iteration reassembles exactly, and never splits a scalar.
    {
        var rebuilt: std.ArrayList(u8) = .empty;
        defer rebuilt.deinit(gpa);
        var chunks = r.slice(.{ .start = 0, .end = ref.len });
        while (chunks.next()) |c| {
            try t.expect(std.unicode.utf8ValidateSlice(c));
            try rebuilt.appendSlice(gpa, c);
        }
        try t.expectEqualStrings(ref, rebuilt.items);
    }

    // Cursor scalar walk matches std.unicode iteration.
    {
        var cur = r.cursorAt(0);
        var view = (try std.unicode.Utf8View.init(ref)).iterator();
        while (view.nextCodepoint()) |expected| {
            try t.expectEqual(@as(?u21, expected), cur.next());
        }
        try t.expectEqual(@as(?u21, null), cur.next());
    }
}

fn runOracle(comptime RopeT: type, seed: u64, op_count: usize) !void {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    var ref: std.ArrayList(u8) = .empty;
    defer ref.deinit(gpa);
    var r: RopeT = .empty;
    defer r.deinit(gpa);

    var snap: ?RopeT = null;
    var frozen: std.ArrayList(u8) = .empty;
    defer frozen.deinit(gpa);
    defer if (snap) |*s| s.deinit(gpa);

    var op: usize = 0;
    while (op < op_count) : (op += 1) {
        var bounds = try refBoundaries(gpa, ref.items);
        defer bounds.deinit(gpa);
        const pick = struct {
            fn pick(rand: std.Random, b: []const usize) usize {
                return b[rand.uintLessThan(usize, b.len)];
            }
        }.pick;

        switch (random.uintLessThan(u8, 10)) {
            // insert (weighted heaviest, like real editing)
            0, 1, 2, 3, 4 => {
                const text = pool[random.uintLessThan(usize, pool.len)];
                const off = pick(random, bounds.items);
                const edit = try r.insert(gpa, off, text);
                try ref.insertSlice(gpa, off, text);
                try t.expectEqual(off, edit.offset);
                try t.expectEqual(@as(usize, 0), edit.removed);
                try t.expectEqual(text.len, edit.inserted);
            },
            // delete
            5, 6, 7 => {
                const a = pick(random, bounds.items);
                const b = pick(random, bounds.items);
                const range: Range = .{ .start = @min(a, b), .end = @max(a, b) };
                const edit = try r.delete(gpa, range);
                try ref.replaceRange(gpa, range.start, range.len(), &.{});
                try t.expectEqual(range.len(), edit.removed);
            },
            // replace
            8 => {
                const a = pick(random, bounds.items);
                const b = pick(random, bounds.items);
                const range: Range = .{ .start = @min(a, b), .end = @max(a, b) };
                const text = pool[random.uintLessThan(usize, pool.len)];
                _ = try r.replace(gpa, range, text);
                try ref.replaceRange(gpa, range.start, range.len(), text);
            },
            // split + re-append: structural ops must round-trip
            9 => {
                const off = pick(random, bounds.items);
                var right = try r.split(gpa, off);
                try t.expectEqual(off, r.byteLen());
                try t.expectEqual(ref.items.len - off, right.byteLen());
                try r.append(gpa, &right);
                try t.expect(right.isEmpty());
            },
            else => unreachable,
        }

        try t.expectEqual(ref.items.len, r.byteLen());

        // Periodically deep-check, and exercise snapshot immutability.
        if (op % 29 == 0) try checkAgainstReference(&r, ref.items);
        if (op % 41 == 0) {
            if (snap) |*s| {
                try checkAgainstReference(s, frozen.items);
                s.deinit(gpa);
            }
            snap = r.snapshot();
            frozen.clearRetainingCapacity();
            try frozen.appendSlice(gpa, ref.items);
        }
    }
    try checkAgainstReference(&r, ref.items);
    if (snap) |*s| try checkAgainstReference(s, frozen.items);
}

test "oracle: default rope, several seeds" {
    try runOracle(Rope, 0xdead_beef, 240);
    try runOracle(Rope, 42, 240);
}

test "oracle: tiny nodes force deep trees" {
    try runOracle(TinyRope, 0xfeed_face, 400);
    try runOracle(TinyRope, 7, 400);
}

test "empty rope basics" {
    const gpa = t.allocator;
    var r: Rope = .empty;
    defer r.deinit(gpa);
    try t.expect(r.isEmpty());
    try t.expectEqual(@as(usize, 0), r.byteLen());
    try t.expectEqual(@as(usize, 1), r.lineCount());
    try t.expectEqual(Point{ .row = 0, .col = 0 }, r.offsetToPoint(0));
    var cur = r.cursorAt(0);
    try t.expectEqual(@as(?u21, null), cur.next());
    _ = try r.insert(gpa, 0, "hello");
    try t.expectEqual(@as(usize, 5), r.byteLen());
    _ = try r.delete(gpa, .{ .start = 0, .end = 5 });
    try t.expect(r.isEmpty());
}

test "fromSlice round-trips and checks against reference" {
    const gpa = t.allocator;
    const text = "The quick brown 狐 jumps\nover the lazy 犬.\n𝄞 plays\na € tune.";
    var r = try Rope.fromSlice(gpa, text);
    defer r.deinit(gpa);
    try checkAgainstReference(&r, text);

    var tiny = try TinyRope.fromSlice(gpa, text);
    defer tiny.deinit(gpa);
    try t.expectEqual(text.len, tiny.byteLen());
    const got = try tiny.toOwnedSlice(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings(text, got);
}

test "fromBacking borrows without copying until edited" {
    const gpa = t.allocator;
    const backing = "line one\nline two\nline three, quite a bit longer\n";
    var r = try TinyRope.fromBacking(gpa, backing);
    defer r.deinit(gpa);

    // Zero-copy: the first chunk aliases the backing store.
    var chunks = r.slice(.{ .start = 0, .end = r.byteLen() });
    const first = chunks.next().?;
    try t.expect(@intFromPtr(first.ptr) >= @intFromPtr(backing.ptr));
    try t.expect(@intFromPtr(first.ptr) < @intFromPtr(backing.ptr) + backing.len);

    // A snapshot pins the pre-edit state; edits splinter borrowed leaves.
    var snap = r.snapshot();
    defer snap.deinit(gpa);
    _ = try r.insert(gpa, 9, "inserted ");
    _ = try r.delete(gpa, .{ .start = 0, .end = 5 });

    const edited = try r.toOwnedSlice(gpa);
    defer gpa.free(edited);
    try t.expectEqualStrings("one\ninserted line two\nline three, quite a bit longer\n", edited);
    const original = try snap.toOwnedSlice(gpa);
    defer gpa.free(original);
    try t.expectEqualStrings(backing, original);
}

test "fromReader streams across chunk boundaries" {
    const gpa = t.allocator;
    // Long enough to span many chunks; multibyte scalars land on read seams.
    var big: std.ArrayList(u8) = .empty;
    defer big.deinit(gpa);
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        try big.appendSlice(gpa, "αβγδε 12345\n日本語のテキスト …\n");
    }
    var reader = std.Io.Reader.fixed(big.items);
    var r = try Rope.fromReader(gpa, &reader);
    defer r.deinit(gpa);
    const got = try r.toOwnedSlice(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings(big.items, got);
}

test "writeTo streams the exact contents" {
    const gpa = t.allocator;
    const text = "stream me 出力 please\n";
    var r = try TinyRope.fromSlice(gpa, text);
    defer r.deinit(gpa);
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try r.writeTo(&w);
    try t.expectEqualStrings(text, w.buffered());
}

test "byteAt and cursor seek/point/utf16" {
    const gpa = t.allocator;
    const text = "ab\ncd €uro\nfin";
    var r = try TinyRope.fromSlice(gpa, text);
    defer r.deinit(gpa);
    for (text, 0..) |b, i| try t.expectEqual(b, r.byteAt(i));

    var cur = r.cursorAt(3); // start of "cd"
    try t.expectEqual(@as(usize, 3), cur.byteOffset());
    try t.expectEqual(Point{ .row = 1, .col = 0 }, cur.point());
    try t.expectEqual(@as(?u21, 'c'), cur.next());
    cur.seekTo(6); // the €
    try t.expectEqual(@as(?u21, '€'), cur.next());
    try t.expectEqual(try std.unicode.calcUtf16LeLen(text[0..cur.byteOffset()]), cur.utf16Offset());
}

test "snapshot shares structure at O(1) and diverges on edit" {
    const gpa = t.allocator;
    var r = try Rope.fromSlice(gpa, "shared until written");
    defer r.deinit(gpa);
    var snap = r.snapshot();
    defer snap.deinit(gpa);
    // Same root pointer: nothing was copied.
    try t.expectEqual(r.root, snap.root);
    _ = try r.replace(gpa, .{ .start = 0, .end = 6 }, "copied");
    const a = try r.toOwnedSlice(gpa);
    defer gpa.free(a);
    const b = try snap.toOwnedSlice(gpa);
    defer gpa.free(b);
    try t.expectEqualStrings("copied until written", a);
    try t.expectEqualStrings("shared until written", b);
}

test "split/append at extremes" {
    const gpa = t.allocator;
    var r = try TinyRope.fromSlice(gpa, "0123456789");
    defer r.deinit(gpa);

    var right = try r.split(gpa, 10); // split at end → empty right
    try t.expect(right.isEmpty());
    try r.append(gpa, &right);

    var right2 = try r.split(gpa, 0); // split at start → empty left
    defer right2.deinit(gpa);
    try t.expect(r.isEmpty());
    try t.expectEqual(@as(usize, 10), right2.byteLen());

    // Append into empty.
    try r.append(gpa, &right2);
    try t.expectEqual(@as(usize, 10), r.byteLen());
}

test "dimension-stripped instantiation compiles and works" {
    const gpa = t.allocator;
    const Bare = rope_mod.RopeWith(.{
        .track_scalars = false,
        .track_utf16 = false,
        .track_lines = false,
        .thread_safe = false,
    });
    var r = try Bare.fromSlice(gpa, "no metrics but bytes\n");
    defer r.deinit(gpa);
    try t.expectEqual(@as(usize, 21), r.byteLen());
    _ = try r.insert(gpa, 0, "…");
    const got = try r.toOwnedSlice(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings("…no metrics but bytes\n", got);
}
