//! stemma benchmark harness.
//!
//! Reports per-work-unit medians over blocks so a regression names its stage:
//! ns/insert for the keystroke path, ns/query for conversions, GB/s for scans
//! and loads. Deterministic PRNG workloads; always built ReleaseFast by the
//! `bench` step regardless of -Doptimize.
//!
//! Usage: zig build bench            (default instantiation only)
//!        zig build bench -- <filter>  (also sweeps knob instantiations)
//! Output: one line per benchmark: `name  median  (best)  n=blocks`.

const std = @import("std");
const Io = std.Io;
const stemma = @import("stemma");

const Rope = stemma.Rope;

const blocks = 11; // odd → clean median
const gpa = std.heap.smp_allocator;

const Timer = struct {
    io: Io,
    start: i96,

    fn start_(io: Io) Timer {
        return .{ .io = io, .start = Io.Clock.now(.awake, io).nanoseconds };
    }

    fn lap(t: *Timer) u64 {
        const now = Io.Clock.now(.awake, t.io).nanoseconds;
        defer t.start = now;
        return @intCast(now - t.start);
    }
};

fn report(w: *Io.Writer, name: []const u8, samples: []u64, per_block_units: usize, comptime unit: []const u8) !void {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const median = samples[samples.len / 2] / per_block_units;
    const best = samples[0] / per_block_units;
    try w.print("{s:<28} {d:>8} {s}/op  (best {d:>8})  n={d}\n", .{ name, median, unit, best, samples.len });
    try w.flush();
}

fn reportRate(w: *Io.Writer, name: []const u8, samples: []u64, bytes_per_block: usize) !void {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const gbs = struct {
        fn gbs(bytes: usize, ns: u64) f64 {
            return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(ns));
        }
    }.gbs;
    try w.print("{s:<28} {d:>8.3} GB/s   (best {d:>8.3})  n={d}\n", .{ name, gbs(bytes_per_block, samples[samples.len / 2]), gbs(bytes_per_block, samples[0]), samples.len });
    try w.flush();
}

/// ~`size` bytes of deterministic ASCII text in 20–100 char lines. Every
/// offset is a scalar boundary, so random offsets need no snapping (keeps
/// offset generation out of the measured op).
fn asciiDoc(size: usize) ![]u8 {
    var text = try std.ArrayList(u8).initCapacity(gpa, size + 128);
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    while (text.items.len < size) {
        const line_len = 20 + random.uintLessThan(usize, 80);
        for (0..line_len) |_| {
            text.appendAssumeCapacity('a' + @as(u8, @intCast(random.uintLessThan(u8, 26))));
        }
        text.appendAssumeCapacity('\n');
        try text.ensureUnusedCapacity(gpa, 128);
    }
    return text.toOwnedSlice(gpa);
}

const typing_pool = [_][]const u8{ "a", "b", " ", "th", "e", "\n", "x", "qu" };
const typing_pool_unicode = [_][]const u8{ "a", "é", " ", "th", "日", "\n", "€", "𝄞" };

/// Sequential typing at an advancing cursor; the uniqueness fast path.
fn benchTyping(comptime RopeT: type, io: Io, w: *Io.Writer, name: []const u8, pool: []const []const u8, with_snapshot: bool) !void {
    const ops_per_block = 100_000;
    var samples: [blocks]u64 = undefined;
    for (&samples) |*s| {
        var r: RopeT = .empty;
        defer r.deinit(gpa);
        var snap: ?RopeT = null;
        defer if (snap) |*sn| sn.deinit(gpa);
        var pos: usize = 0;
        var prng = std.Random.DefaultPrng.init(0xbead);
        const random = prng.random();
        var timer = Timer.start_(io);
        for (0..ops_per_block) |i| {
            const text = pool[i % pool.len];
            _ = try r.insert(gpa, pos, text);
            pos += text.len;
            if (i % 1024 == 0) {
                // Occasional cursor jump, like a human relocating.
                pos = r.scalarToOffset(random.uintLessThan(usize, r.scalarLen() + 1));
                if (with_snapshot) {
                    if (snap) |*sn| sn.deinit(gpa);
                    snap = r.snapshot();
                }
            }
        }
        s.* = timer.lap();
        std.mem.doNotOptimizeAway(r.byteLen());
    }
    try report(w, name, &samples, ops_per_block, "ns");
}

/// Random insert/delete churn in a steady ~1 MiB ASCII document.
fn benchRandomEdit(comptime RopeT: type, io: Io, w: *Io.Writer, name: []const u8) !void {
    const doc = try asciiDoc(1 << 20);
    defer gpa.free(doc);
    const ops_per_block = 20_000;
    var samples: [blocks]u64 = undefined;
    for (&samples) |*s| {
        var r = try RopeT.fromSlice(gpa, doc);
        defer r.deinit(gpa);
        var prng = std.Random.DefaultPrng.init(0xd1ce);
        const random = prng.random();
        var timer = Timer.start_(io);
        for (0..ops_per_block) |i| {
            const len = r.byteLen();
            if (i % 2 == 0) {
                _ = try r.insert(gpa, random.uintLessThan(usize, len + 1), "hello world ");
            } else {
                const start = random.uintLessThan(usize, len - 12);
                _ = try r.delete(gpa, .{ .start = start, .end = start + 12 });
            }
        }
        s.* = timer.lap();
        std.mem.doNotOptimizeAway(r.byteLen());
    }
    try report(w, name, &samples, ops_per_block, "ns");
}

fn benchConversions(comptime RopeT: type, io: Io, w: *Io.Writer, name_prefix: []const u8) !void {
    const doc = try asciiDoc(8 << 20);
    defer gpa.free(doc);
    var r = try RopeT.fromSlice(gpa, doc);
    defer r.deinit(gpa);
    const queries_per_block = 200_000;

    var name_buf: [64]u8 = undefined;
    inline for (.{ "offsetToPoint", "pointToOffset", "offsetToUtf16" }) |which| {
        var samples: [blocks]u64 = undefined;
        for (&samples) |*s| {
            var prng = std.Random.DefaultPrng.init(0xcafe);
            const random = prng.random();
            var sink: usize = 0;
            var timer = Timer.start_(io);
            for (0..queries_per_block) |_| {
                const off = random.uintLessThan(usize, r.byteLen() + 1);
                if (comptime std.mem.eql(u8, which, "offsetToPoint")) {
                    sink +%= r.offsetToPoint(off).col;
                } else if (comptime std.mem.eql(u8, which, "pointToOffset")) {
                    const p = r.offsetToPoint(off);
                    sink +%= r.pointToOffset(p);
                } else {
                    sink +%= r.offsetToUtf16(off);
                }
            }
            s.* = timer.lap();
            std.mem.doNotOptimizeAway(sink);
        }
        const label = try std.fmt.bufPrint(&name_buf, "{s}{s}", .{ name_prefix, which });
        try report(w, label, &samples, queries_per_block, "ns");
    }
}

fn benchLoadAndScan(comptime RopeT: type, io: Io, w: *Io.Writer, name_prefix: []const u8) !void {
    const size = 32 << 20;
    const doc = try asciiDoc(size);
    defer gpa.free(doc);
    var name_buf: [64]u8 = undefined;

    {
        var samples: [blocks]u64 = undefined;
        for (&samples) |*s| {
            var timer = Timer.start_(io);
            var r = try RopeT.fromSlice(gpa, doc);
            s.* = timer.lap();
            r.deinit(gpa);
        }
        try reportRate(w, try std.fmt.bufPrint(&name_buf, "{s}fromSlice 32MiB", .{name_prefix}), &samples, doc.len);
    }
    {
        var samples: [blocks]u64 = undefined;
        for (&samples) |*s| {
            var timer = Timer.start_(io);
            var r = try RopeT.fromBacking(gpa, doc);
            s.* = timer.lap();
            r.deinit(gpa);
        }
        try reportRate(w, try std.fmt.bufPrint(&name_buf, "{s}fromBacking 32MiB", .{name_prefix}), &samples, doc.len);
    }
    {
        var r = try RopeT.fromSlice(gpa, doc);
        defer r.deinit(gpa);
        var samples: [blocks]u64 = undefined;
        for (&samples) |*s| {
            var sink: usize = 0;
            var timer = Timer.start_(io);
            var chunks = r.slice(.{ .start = 0, .end = r.byteLen() });
            while (chunks.next()) |c| sink +%= c.len;
            s.* = timer.lap();
            std.mem.doNotOptimizeAway(sink);
        }
        try reportRate(w, try std.fmt.bufPrint(&name_buf, "{s}chunk scan 32MiB", .{name_prefix}), &samples, doc.len);
    }
    {
        var r = try RopeT.fromSlice(gpa, doc);
        defer r.deinit(gpa);
        const snaps_per_block = 1000;
        var samples: [blocks]u64 = undefined;
        for (&samples) |*s| {
            var timer = Timer.start_(io);
            for (0..snaps_per_block) |_| {
                var snap = r.snapshot();
                std.mem.doNotOptimizeAway(snap.byteLen());
                snap.deinit(gpa);
            }
            s.* = timer.lap();
        }
        try report(w, try std.fmt.bufPrint(&name_buf, "{s}snapshot+drop", .{name_prefix}), &samples, snaps_per_block, "ns");
    }
}

/// Collaboration-layer benchmarks: the collab tax on local typing, and merge
/// throughput (v1 replays from genesis — these numbers are the baseline the
/// optimization ladder in BENCHMARKS.md is measured against).
fn benchCollab(io: Io, w: *Io.Writer) !void {
    const TextDoc = stemma.TextDoc;

    // Local typing through TextDoc (event recording + rope) vs bare rope.
    {
        const ops_per_block = 20_000;
        var samples: [blocks]u64 = undefined;
        for (&samples) |*s| {
            var d: TextDoc = .empty;
            defer d.deinit(gpa);
            try d.setAgent(gpa, "bench");
            var pos: usize = 0;
            var timer = Timer.start_(io);
            for (0..ops_per_block) |i| {
                const text = typing_pool[i % typing_pool.len];
                _ = try d.insert(gpa, pos, text);
                pos += text.len;
            }
            s.* = timer.lap();
            std.mem.doNotOptimizeAway(d.text().byteLen());
        }
        try report(w, "collab doc-typing ascii", &samples, ops_per_block, "ns");
    }
    // Merge a linear 4k-unit document into an empty doc (open()).
    {
        var author: TextDoc = .empty;
        defer author.deinit(gpa);
        try author.setAgent(gpa, "author");
        for (0..4096) |i| {
            _ = try author.insert(gpa, author.text().byteLen(), if (i % 64 == 63) "\n" else "x");
        }
        const bytes = try author.serialize(gpa);
        defer gpa.free(bytes);
        const merge_blocks = 5;
        var samples: [merge_blocks]u64 = undefined;
        for (&samples) |*s| {
            var timer = Timer.start_(io);
            var d = try TextDoc.open(gpa, bytes);
            s.* = timer.lap();
            d.deinit(gpa);
        }
        try report(w, "collab merge linear 4k units", &samples, 4096, "ns");
    }
    // Cross-merge two concurrent 1k-unit branches.
    {
        const merge_blocks = 5;
        var samples: [merge_blocks]u64 = undefined;
        for (&samples) |*s| {
            var alice: TextDoc = .empty;
            defer alice.deinit(gpa);
            var bob: TextDoc = .empty;
            defer bob.deinit(gpa);
            try alice.setAgent(gpa, "alice");
            try bob.setAgent(gpa, "bob");
            _ = try alice.insert(gpa, 0, "base ");
            const base = try alice.serialize(gpa);
            defer gpa.free(base);
            gpa.free(try bob.merge(gpa, base));
            for (0..1024) |_| {
                _ = try alice.insert(gpa, alice.text().byteLen(), "a");
                _ = try bob.insert(gpa, 5, "b");
            }
            const vb = try bob.version(gpa);
            defer gpa.free(vb);
            const batch = try alice.eventsSince(gpa, vb);
            defer gpa.free(batch);
            var timer = Timer.start_(io);
            const edits = try bob.merge(gpa, batch);
            s.* = timer.lap();
            gpa.free(edits);
            std.mem.doNotOptimizeAway(bob.text().byteLen());
        }
        try report(w, "collab merge concurrent 1k+1k", &samples, 1024, "ns");
    }
}

fn runSuite(comptime RopeT: type, io: Io, w: *Io.Writer, comptime prefix: []const u8, filter: ?[]const u8) !void {
    const want = struct {
        fn want(f: ?[]const u8, name: []const u8) bool {
            return f == null or std.mem.indexOf(u8, name, f.?) != null;
        }
    }.want;
    if (want(filter, prefix ++ "typing")) {
        try benchTyping(RopeT, io, w, prefix ++ "typing ascii", &typing_pool, false);
        try benchTyping(RopeT, io, w, prefix ++ "typing unicode", &typing_pool_unicode, false);
        try benchTyping(RopeT, io, w, prefix ++ "typing ascii +snap", &typing_pool, true);
    }
    if (want(filter, prefix ++ "random-edit")) try benchRandomEdit(RopeT, io, w, prefix ++ "random-edit 1MiB");
    if (want(filter, prefix ++ "conv")) try benchConversions(RopeT, io, w, prefix ++ "conv ");
    if (want(filter, prefix ++ "load")) try benchLoadAndScan(RopeT, io, w, prefix ++ "load ");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const filter: ?[]const u8 = if (args.len > 1) args[1] else null;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const w = &stdout_writer.interface;

    // Default instantiation is the headline; with a filter, knob variants
    // answer the chunk-capacity / branch-factor question on the same
    // workloads.
    try runSuite(Rope, io, w, "", filter);
    if (filter == null or std.mem.indexOf(u8, "collab", filter.?) != null) try benchCollab(io, w);
    if (filter != null) {
        try runSuite(stemma.RopeWith(.{ .chunk_capacity = 128 }), io, w, "c128/", filter);
        try runSuite(stemma.RopeWith(.{ .chunk_capacity = 512 }), io, w, "c512/", filter);
        try runSuite(stemma.RopeWith(.{ .chunk_capacity = 512, .branch_factor = 16 }), io, w, "c512b16/", filter);
        try runSuite(stemma.RopeWith(.{ .chunk_capacity = 1024, .branch_factor = 16 }), io, w, "c1024b16/", filter);
        try runSuite(stemma.RopeWith(.{ .branch_factor = 4 }), io, w, "b4/", filter);
        try runSuite(stemma.RopeWith(.{ .branch_factor = 16 }), io, w, "b16/", filter);
        try runSuite(stemma.RopeWith(.{ .thread_safe = false }), io, w, "noatomic/", filter);
    }
    try w.flush();
}
