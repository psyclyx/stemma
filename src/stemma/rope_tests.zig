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
    var row: usize = 0;
    var line_start: usize = 0;
    for (bytes[0..off], 0..) |b, i| {
        if (b == '\n') {
            row += 1;
            line_start = i + 1;
        }
    }
    return .{ .row = row, .col = off - line_start };
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

    // Cursor scalar walk matches std.unicode iteration — and walking back
    // from the end reproduces it in reverse, landing exactly at 0.
    {
        var cur = r.cursorAt(0);
        var view = (try std.unicode.Utf8View.init(ref)).iterator();
        while (view.nextCodepoint()) |expected| {
            try t.expectEqual(@as(?u21, expected), cur.next());
        }
        try t.expectEqual(@as(?u21, null), cur.next());

        var fwd: std.ArrayList(u21) = .empty;
        defer fwd.deinit(gpa);
        var view2 = (try std.unicode.Utf8View.init(ref)).iterator();
        while (view2.nextCodepoint()) |cp| try fwd.append(gpa, cp);
        var back = r.cursorAt(ref.len);
        var remaining = fwd.items.len;
        while (back.prev()) |cp| {
            remaining -= 1;
            try t.expectEqual(fwd.items[remaining], cp);
        }
        try t.expectEqual(@as(usize, 0), remaining);
        try t.expectEqual(@as(usize, 0), back.byteOffset());
    }

    // Reverse chunk walk reassembles the contents.
    {
        var rebuilt: std.ArrayList(u8) = .empty;
        defer rebuilt.deinit(gpa);
        var cur = r.cursorAt(ref.len);
        while (cur.prevChunk()) |c| try rebuilt.insertSlice(gpa, 0, c);
        try t.expectEqualStrings(ref, rebuilt.items);
    }

    // Structural invariants hold.
    r.validate();
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

test "eql: sharing, structure-independence, and difference" {
    const gpa = t.allocator;
    var a = try TinyRope.fromSlice(gpa, "same content, different trees\n");
    defer a.deinit(gpa);

    // Snapshot shares the root: O(1) true.
    var snap = a.snapshot();
    defer snap.deinit(gpa);
    try t.expect(a.eql(snap));

    // Same content built by a different edit history: still equal.
    var b = try TinyRope.fromSlice(gpa, "different trees\n");
    defer b.deinit(gpa);
    _ = try b.insert(gpa, 0, "same content, ");
    try t.expect(a.eql(b));
    try t.expect(b.eql(a));

    // One byte of difference: not equal.
    _ = try b.replace(gpa, .{ .start = 0, .end = 4 }, "sane");
    try t.expect(!a.eql(b));

    // Length short-circuit.
    _ = try b.delete(gpa, .{ .start = 0, .end = 5 });
    try t.expect(!a.eql(b));

    var e1: TinyRope = .empty;
    var e2: TinyRope = .empty;
    try t.expect(e1.eql(e2));
    try t.expect(!e1.eql(a));
    _ = &e1;
    _ = &e2;
}

test "zigzag cursor: interleaved next/prev tracks reference" {
    const gpa = t.allocator;
    const text = "aé€𝄞\nbcℝ\nxyz日本語 end";
    var r = try TinyRope.fromSlice(gpa, text);
    defer r.deinit(gpa);

    var prng = std.Random.DefaultPrng.init(0x2162);
    const random = prng.random();
    var cur = r.cursorAt(0);
    var off: usize = 0;
    for (0..300) |_| {
        if (random.boolean()) {
            if (cur.next()) |cp| {
                off += std.unicode.utf8CodepointSequenceLength(cp) catch unreachable;
            } else try t.expectEqual(text.len, off);
        } else {
            if (cur.prev()) |cp| {
                off -= std.unicode.utf8CodepointSequenceLength(cp) catch unreachable;
                // The scalar we stepped back over starts at the new offset.
                const l = std.unicode.utf8ByteSequenceLength(text[off]) catch unreachable;
                try t.expectEqual(std.unicode.utf8Decode(text[off..][0..l]) catch unreachable, cp);
            } else try t.expectEqual(@as(usize, 0), off);
        }
        try t.expectEqual(off, cur.byteOffset());
    }
}

test "streamReader: ranged, chunked, Reader-API compatible" {
    const gpa = t.allocator;
    const text = "stream a rope through std.Io.Reader — even 日本語 survives chunking";
    var r = try TinyRope.fromSlice(gpa, text);
    defer r.deinit(gpa);

    // Whole-buffer streaming into a fixed writer.
    {
        var sr = r.streamReader(.{ .start = 0, .end = r.byteLen() }, &.{});
        var out: [256]u8 = undefined;
        var w = std.Io.Writer.fixed(&out);
        const n = try sr.interface.streamRemaining(&w);
        try t.expectEqual(text.len, n);
        try t.expectEqualStrings(text, w.buffered());
    }
    // Sub-range via readSliceAll (exercises the readVec → stream path).
    {
        var sr = r.streamReader(.{ .start = 7, .end = 13 }, &.{});
        var out: [6]u8 = undefined;
        try sr.interface.readSliceAll(&out);
        try t.expectEqualStrings("a rope", &out);
        try t.expectError(error.EndOfStream, sr.interface.readSliceAll(&out));
    }
}

// ── Serious-editor hardening ────────────────────────────────────────────

/// Content equality without allocating (safe under a failing allocator).
fn expectContent(r: anytype, expected: []const u8) !void {
    try t.expectEqual(expected.len, r.byteLen());
    var i: usize = 0;
    var chunks = r.slice(.{ .start = 0, .end = expected.len });
    while (chunks.next()) |c| {
        try t.expect(std.mem.eql(u8, expected[i..][0..c.len], c));
        i += c.len;
    }
    try t.expectEqual(expected.len, i);
}

/// Scripted general-path sequence for allocation-fault injection. Every
/// alloc-failing op must leave the rope (and a live snapshot) byte-identical
/// to its pre-op state before propagating OOM; every node must be freed
/// afterwards (the harness leak-checks each fail index).
fn oomScript(gpa: std.mem.Allocator) !void {
    const d0 = "abcdefghij" ** 10; // deep tree at chunk_capacity=8
    const ins = "INSERTED_TEXT_LONGER_THAN_LEAF";
    const e1 = d0[0..50] ++ ins ++ d0[50..];
    const e2 = e1[0..40] ++ e1[90..];
    const e3 = e2[0..10] ++ "xyz" ++ e2[20..];

    var r = try TinyRope.fromSlice(gpa, d0);
    defer r.deinit(gpa);

    var snap = r.snapshot();
    defer snap.deinit(gpa);

    _ = r.insert(gpa, 50, ins) catch |err| {
        try expectContent(&r, d0);
        return err;
    };
    try expectContent(&r, e1);

    _ = r.delete(gpa, .{ .start = 40, .end = 90 }) catch |err| {
        try expectContent(&r, e1);
        try expectContent(&snap, d0);
        return err;
    };
    try expectContent(&r, e2);

    _ = r.replace(gpa, .{ .start = 10, .end = 20 }, "xyz") catch |err| {
        try expectContent(&r, e2);
        return err;
    };
    try expectContent(&r, e3);

    var right = r.split(gpa, 30) catch |err| {
        try expectContent(&r, e3);
        return err;
    };
    errdefer right.deinit(gpa);
    try expectContent(&r, e3[0..30]);
    r.append(gpa, &right) catch |err| {
        try expectContent(&r, e3[0..30]);
        try expectContent(&right, e3[30..]);
        return err;
    };
    try expectContent(&r, e3);

    // The snapshot rode through everything untouched.
    try expectContent(&snap, d0);
    r.validate();
    snap.validate();
}

test "OOM: every allocation failure leaves the rope unchanged and leak-free" {
    try std.testing.checkAllAllocationFailures(t.allocator, oomScript, .{});
}

test "snapshots cross threads: readers verify frozen state under live edits" {
    if (@import("builtin").single_threaded) return error.SkipZigTest;
    // testing.allocator is not thread-safe; snapshots deinit on other threads.
    const A = std.heap.smp_allocator;

    const ReaderThread = struct {
        fn run(snap: Rope, expected: []const u8) void {
            var s = snap;
            defer s.deinit(std.heap.smp_allocator);
            defer std.heap.smp_allocator.free(expected);
            for (0..64) |_| {
                var i: usize = 0;
                var chunks = s.slice(.{ .start = 0, .end = s.byteLen() });
                while (chunks.next()) |c| {
                    std.debug.assert(std.mem.eql(u8, expected[i..][0..c.len], c));
                    i += c.len;
                }
                std.debug.assert(i == expected.len);
                std.debug.assert(s.offsetToPoint(i / 2).row <= s.lineCount());
            }
        }
    };

    var r = try Rope.fromSlice(A, "base line\n" ** 40);
    defer r.deinit(A);

    var threads: [6]std.Thread = undefined;
    var spawned: usize = 0;
    defer for (threads[0..spawned]) |th| th.join();
    for (&threads) |*th| {
        // Edit between spawns so every snapshot freezes a different state
        // and the writer keeps mutating while readers verify.
        for (0..120) |k| {
            _ = try r.insert(A, r.byteLen() / 2, "concurrent edit ");
            if (k % 3 == 0) {
                const start = r.byteLen() / 3;
                _ = try r.delete(A, .{ .start = start, .end = start + 8 });
            }
        }
        const snap = r.snapshot();
        const expected = try r.toOwnedSlice(A);
        th.* = try std.Thread.spawn(.{}, ReaderThread.run, .{ snap, expected });
        spawned += 1;
    }
}

fn fuzzOps(_: void, smith: *std.testing.Smith) !void {
    const gpa = t.allocator;
    var ref: std.ArrayList(u8) = .empty;
    defer ref.deinit(gpa);
    var r: TinyRope = .empty;
    defer r.deinit(gpa);

    var ops: usize = 0;
    while (ops < 512 and !smith.eosWithHash(0x5731)) : (ops += 1) {
        var bounds = try refBoundaries(gpa, ref.items);
        defer bounds.deinit(gpa);
        const a = bounds.items[smith.indexWithHash(bounds.items.len, 0x0a11)];
        const b = bounds.items[smith.indexWithHash(bounds.items.len, 0x0b22)];
        const range: Range = .{ .start = @min(a, b), .end = @max(a, b) };
        const text = pool[smith.indexWithHash(pool.len, 0x7e37)];
        switch (smith.valueRangeLessThanWithHash(u8, 0, 4, 0x0004)) {
            0 => {
                _ = try r.insert(gpa, a, text);
                try ref.insertSlice(gpa, a, text);
            },
            1 => {
                _ = try r.delete(gpa, range);
                try ref.replaceRange(gpa, range.start, range.len(), &.{});
            },
            2 => {
                _ = try r.replace(gpa, range, text);
                try ref.replaceRange(gpa, range.start, range.len(), text);
            },
            3 => {
                var right = try r.split(gpa, a);
                try r.append(gpa, &right);
            },
            else => unreachable,
        }
        try t.expectEqual(ref.items.len, r.byteLen());
    }
    try checkAgainstReference(&r, ref.items);
}

test "fuzz: smith-driven op stream against the oracle" {
    try std.testing.fuzz({}, fuzzOps, .{});
}

test "large borrowed backing: many spans, cross-leaf edits, invariants" {
    const gpa = t.allocator;
    const Borrow64K = rope_mod.RopeWith(.{ .borrowed_capacity = 1 << 16 });

    // 8 MiB deterministic doc, newline every 64 bytes → 128 borrowed spans.
    const size = 8 << 20;
    const doc = try gpa.alloc(u8, size);
    defer gpa.free(doc);
    for (doc, 0..) |*byte, i| {
        byte.* = if (i % 64 == 63) '\n' else 'a' + @as(u8, @intCast(i % 26));
    }

    var r = try Borrow64K.fromBacking(gpa, doc);
    defer r.deinit(gpa);
    try t.expectEqual(size, r.byteLen());
    try t.expectEqual(size / 64 + 1, r.lineCount());
    try t.expectEqual(Point{ .row = 1, .col = 0 }, r.offsetToPoint(64));
    try t.expectEqual(@as(usize, size / 2), r.offsetToScalar(size / 2));

    var snap = r.snapshot();
    defer snap.deinit(gpa);

    // Edits crossing multiple borrowed leaves.
    _ = try r.insert(gpa, size / 2, "spliced into the middle of an mmap span");
    _ = try r.delete(gpa, .{ .start = 1 << 16, .end = (1 << 16) * 3 }); // two whole spans + boundary
    _ = try r.replace(gpa, .{ .start = 0, .end = 128 }, "rewritten head\n");
    r.validate();

    // Snapshot still equals the pristine backing; rope diverged.
    try expectContent(&snap, doc);
    try t.expect(!r.eql(snap));
}

test "find/findLast property: matches std.mem on flattened content" {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(0xf19d);
    const random = prng.random();

    var doc: std.ArrayList(u8) = .empty;
    defer doc.deinit(gpa);
    while (doc.items.len < 2048) {
        try doc.appendSlice(gpa, pool[random.uintLessThan(usize, pool.len)]);
    }
    // chunk_capacity=8 → needles longer than 8 bytes always straddle leaves.
    var r = try TinyRope.fromSlice(gpa, doc.items);
    defer r.deinit(gpa);
    const full: Range = .{ .start = 0, .end = doc.items.len };

    for (0..200) |_| {
        const a = random.uintLessThan(usize, doc.items.len - 24);
        const nlen = 1 + random.uintLessThan(usize, 20);
        const needle = doc.items[a..][0..nlen];
        try t.expectEqual(std.mem.indexOf(u8, doc.items, needle), r.find(full, needle));
        try t.expectEqual(std.mem.lastIndexOf(u8, doc.items, needle), r.findLast(full, needle));

        // Sub-range agreement.
        const lo = random.uintLessThan(usize, doc.items.len / 2);
        const hi = lo + random.uintLessThan(usize, doc.items.len - lo);
        const sub: Range = .{ .start = lo, .end = hi };
        const expect_sub: ?usize = if (std.mem.indexOfPos(u8, doc.items[0..hi], lo, needle)) |p| p else null;
        try t.expectEqual(expect_sub, r.find(sub, needle));
    }

    try t.expectEqual(@as(?usize, null), r.find(full, "~~~not present~~~"));

    // Iterator finds every non-overlapping occurrence.
    var n_std: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, doc.items, pos, "he")) |p| {
        n_std += 1;
        pos = p + 2;
    }
    var n_rope: usize = 0;
    var it = r.findIterator(full, "he");
    while (it.next()) |_| n_rope += 1;
    try t.expectEqual(n_std, n_rope);
}

test "lineIterator agrees with lineRange" {
    const gpa = t.allocator;
    const text = "first\nsecond line\n\nfourth 𝄞\nlast without newline";
    var r = try TinyRope.fromSlice(gpa, text);
    defer r.deinit(gpa);

    var lines = r.lineIterator(0);
    var row: usize = 0;
    while (lines.next()) |range| : (row += 1) {
        const expected = r.lineRange(row);
        try t.expectEqual(expected, range);
    }
    try t.expectEqual(r.lineCount(), row);

    // Starting mid-document.
    var from2 = r.lineIterator(2);
    try t.expectEqual(r.lineRange(2), from2.next().?);
    try t.expectEqual(r.lineRange(3), from2.next().?);
}

test "PointUtf16 conversions vs std.unicode reference" {
    const gpa = t.allocator;
    const text = "ascii row\n𝄞𝄞 astral\nmixed é€ạ row\n";
    var r = try TinyRope.fromSlice(gpa, text);
    defer r.deinit(gpa);

    var bounds = try refBoundaries(gpa, text);
    defer bounds.deinit(gpa);
    for (bounds.items) |off| {
        const p = r.offsetToPointUtf16(off);
        const bp = refPoint(text, off);
        try t.expectEqual(bp.row, p.row);
        const line_start = off - bp.col;
        try t.expectEqual(try std.unicode.calcUtf16LeLen(text[line_start..off]), p.col);
        try t.expectEqual(off, r.pointUtf16ToOffset(p));
    }
}

// ── Lazy / unrealized content ───────────────────────────────────────────

test "unrealized: out-of-order realization converges to the reference" {
    const gpa = t.allocator;
    // Reference "remote file": realized in scattered windows, both copied
    // and borrowed, until complete.
    var ref: std.ArrayList(u8) = .empty;
    defer ref.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(0x1a2b);
    const random = prng.random();
    while (ref.items.len < 2048) {
        try ref.appendSlice(gpa, pool[random.uintLessThan(usize, pool.len)]);
    }
    const doc = ref.items;

    var r = try TinyRope.fromUnrealized(gpa, doc.len);
    defer r.deinit(gpa);
    try t.expectEqual(doc.len, r.byteLen());
    try t.expect(!r.isRealized(.{ .start = 0, .end = doc.len }));
    try t.expectEqual(@as(usize, 1), r.lineCount()); // no realized newlines yet

    // Window boundaries must be scalar boundaries of the true content.
    var bounds = try refBoundaries(gpa, doc);
    defer bounds.deinit(gpa);
    const cut = struct {
        fn cut(b: []const usize, approx: usize) usize {
            var best: usize = 0;
            for (b) |x| {
                if (x <= approx) best = x else break;
            }
            return best;
        }
    }.cut;
    const c1 = cut(bounds.items, doc.len / 3);
    const c2 = cut(bounds.items, 2 * doc.len / 3);

    // Realize middle first (borrowed), then tail (copied), then head.
    try r.realizeBacking(gpa, c1, doc[c1..c2]);
    try t.expect(r.isRealized(.{ .start = c1, .end = c2 }));
    try t.expect(!r.isRealized(.{ .start = 0, .end = doc.len }));
    {
        // The fetch list is exactly the two remaining runs.
        var it = r.unrealized(.{ .start = 0, .end = doc.len });
        try t.expectEqual(Range{ .start = 0, .end = c1 }, it.next().?);
        try t.expectEqual(Range{ .start = c2, .end = doc.len }, it.next().?);
        try t.expectEqual(@as(?Range, null), it.next());
    }
    try r.realize(gpa, c2, doc[c2..]);
    try r.realize(gpa, 0, doc[0..c1]);
    try t.expect(r.isRealized(.{ .start = 0, .end = doc.len }));

    // Fully realized: contents and every metric match the reference.
    try checkAgainstReference(&r, doc);
}

test "unrealized: byte-domain edits work on and around holes" {
    const gpa = t.allocator;
    var r = try TinyRope.fromUnrealized(gpa, 1000);
    defer r.deinit(gpa);

    // Snapshot before any realization: shares the hole.
    var snap = r.snapshot();
    defer snap.deinit(gpa);

    // Insert INTO the middle of the hole (splits it).
    _ = try r.insert(gpa, 500, "<mark>");
    try t.expectEqual(@as(usize, 1006), r.byteLen());
    try t.expect(r.isRealized(.{ .start = 500, .end = 506 }));

    // Delete a fully-unrealized span (never fetched — pure structure).
    _ = try r.delete(gpa, .{ .start = 0, .end = 100 });
    try t.expectEqual(@as(usize, 906), r.byteLen());

    // Delete spanning unrealized + realized: drops 2 hole bytes + "<m".
    _ = try r.delete(gpa, .{ .start = 398, .end = 402 });
    try t.expectEqual(@as(usize, 902), r.byteLen());

    // Split/append round-trip with holes.
    var right = try r.split(gpa, 400);
    try t.expectEqual(@as(usize, 400), r.byteLen());
    try r.append(gpa, &right);
    try t.expectEqual(@as(usize, 902), r.byteLen());
    r.validate();

    // The snapshot still sees the original untouched hole.
    try t.expectEqual(@as(usize, 1000), snap.byteLen());
    try t.expect(!snap.isRealized(.{ .start = 0, .end = 1000 }));
    snap.validate();

    // Metrics count realized content only: "ark>" survives the deletes.
    try t.expectEqual(@as(usize, 4), r.scalarLen());
}

test "unrealized: partial metrics converge as realization proceeds" {
    const gpa = t.allocator;
    const doc = "line one\nline two\nline three\n";
    var r = try Rope.fromUnrealized(gpa, doc.len);
    defer r.deinit(gpa);

    try t.expectEqual(@as(usize, 1), r.lineCount());
    try r.realize(gpa, 0, doc[0..9]); // "line one\n"
    try t.expectEqual(@as(usize, 2), r.lineCount());
    try t.expectEqual(@as(usize, 9), r.scalarLen());
    try r.realize(gpa, 9, doc[9..]);
    try t.expectEqual(@as(usize, 4), r.lineCount());
    const got = try r.toOwnedSlice(gpa);
    defer gpa.free(got);
    try t.expectEqualStrings(doc, got);
}

fn unrealizedOomScript(gpa: std.mem.Allocator) !void {
    var r = try TinyRope.fromUnrealized(gpa, 64);
    defer r.deinit(gpa);
    try r.realize(gpa, 16, "0123456789abcdef");
    _ = try r.insert(gpa, 32, "xx");
    _ = try r.delete(gpa, .{ .start = 0, .end = 8 });
    try r.realizeBacking(gpa, 40, "ABCDEFGH");
    var right = try r.split(gpa, 30);
    errdefer right.deinit(gpa);
    try r.append(gpa, &right);
    r.validate();
}

test "OOM: unrealized paths are leak-free and restore on failure" {
    try std.testing.checkAllAllocationFailures(t.allocator, unrealizedOomScript, .{});
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
