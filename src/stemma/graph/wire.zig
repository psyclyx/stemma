//! Shared wire-format primitives: unsigned LEB128 over byte slices.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn putUv(gpa: Allocator, out: *std.ArrayList(u8), value: u64) Allocator.Error!void {
    var v = value;
    while (true) {
        const byte: u8 = @intCast(v & 0x7f);
        v >>= 7;
        if (v == 0) {
            try out.append(gpa, byte);
            return;
        }
        try out.append(gpa, byte | 0x80);
    }
}

pub fn getUv(cur: *[]const u8) error{Corrupt}!u64 {
    var result: u64 = 0;
    for (0..10) |i| {
        if (cur.len == 0) return error.Corrupt;
        const byte = cur.*[0];
        cur.* = cur.*[1..];
        const shift: u6 = @intCast(i * 7); // i ≤ 9 → 63
        result |= @as(u64, byte & 0x7f) << shift;
        if (byte & 0x80 == 0) return result;
    }
    return error.Corrupt;
}

pub fn getBytes(cur: *[]const u8, max: usize) error{Corrupt}![]const u8 {
    const len = try getUv(cur);
    if (len > max or len > cur.len) return error.Corrupt;
    const out = cur.*[0..len];
    cur.* = cur.*[len..];
    return out;
}

test "LEB128 roundtrip" {
    const t = std.testing;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    const cases = [_]u64{ 0, 1, 127, 128, 300, 1 << 20, std.math.maxInt(u64) };
    for (cases) |v| try putUv(t.allocator, &buf, v);
    var cur: []const u8 = buf.items;
    for (cases) |v| try t.expectEqual(v, try getUv(&cur));
    try t.expectEqual(@as(usize, 0), cur.len);
}
