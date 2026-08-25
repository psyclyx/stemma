//! Shared document scaffolding: the pieces every collaborative document
//! type implements identically — opaque version tokens, version comparison,
//! and batch agent-table helpers. Doc types pass their `EventGraph`
//! instance (any `Op` payload) as an `anytype` history.

const std = @import("std");
const Allocator = std.mem.Allocator;

const causal = @import("causal.zig");
const wire = @import("wire.zig");
const Lv = causal.Lv;
const AgentId = causal.AgentId;
const putUv = wire.putUv;
const getUv = wire.getUv;
const getBytes = wire.getBytes;

pub const version_magic = "stv\x01";

pub const VersionError = Allocator.Error || error{ Corrupt, MissingDependency };

/// Encode the current frontier as an opaque, replica-portable token
/// (agent NAMES + seqs — local numbering never crosses the wire).
pub fn encodeVersion(gpa: Allocator, history: anytype) Allocator.Error![]u8 {
    const PortableHead = struct {
        name: []const u8,
        seq: u64,
    };

    var heads: std.ArrayList(PortableHead) = .empty;
    defer heads.deinit(gpa);
    try heads.ensureTotalCapacity(gpa, history.frontier.items.len);
    for (history.frontier.items) |lv| {
        const id = history.idOf(lv);
        try heads.append(gpa, .{
            .name = history.agentName(id.agent),
            .seq = id.seq,
        });
    }

    // `Lv` is intentionally replica-local.  The frontier itself may have
    // the same portable heads in different local orders, so the order on
    // the wire must be derived only from the portable identity.
    const PortableOrder = struct {
        fn less(_: void, a: PortableHead, b: PortableHead) bool {
            return switch (std.mem.order(u8, a.name, b.name)) {
                .lt => true,
                .gt => false,
                .eq => a.seq < b.seq,
            };
        }
    };
    std.mem.sort(PortableHead, heads.items, {}, PortableOrder.less);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, version_magic);
    try putUv(gpa, &out, heads.items.len);
    for (heads.items) |head| {
        try putUv(gpa, &out, head.name.len);
        try out.appendSlice(gpa, head.name);
        try putUv(gpa, &out, head.seq);
    }
    return out.toOwnedSlice(gpa);
}

/// Decode a version token to known Lvs. `strict` errors on entries we have
/// not stored (`MissingDependency`); lenient mode skips them.
pub fn decodeVersion(
    history: anytype,
    gpa: Allocator,
    token: []const u8,
    strict: bool,
    out: *std.ArrayList(Lv),
) VersionError!void {
    var cur = token;
    if (!std.mem.startsWith(u8, cur, version_magic)) return error.Corrupt;
    cur = cur[version_magic.len..];
    const count = try getUv(&cur);
    if (count > 1 << 20) return error.Corrupt;
    for (0..count) |_| {
        const name = try getBytes(&cur, 4096);
        if (name.len == 0) return error.Corrupt;
        const seq = try getUv(&cur);
        const lv: ?Lv = if (history.findAgent(name)) |agent|
            history.lvOf(.{ .agent = agent, .seq = seq })
        else
            null;
        if (lv) |v| {
            try out.append(gpa, v);
        } else if (strict) {
            return error.MissingDependency;
        }
    }
}

/// Causal relation between two version tokens (both fully stored locally).
pub fn compareVersions(
    gpa: Allocator,
    history: anytype,
    a_token: []const u8,
    b_token: []const u8,
) VersionError!causal.VersionOrder {
    var a: std.ArrayList(Lv) = .empty;
    defer a.deinit(gpa);
    try decodeVersion(history, gpa, a_token, true, &a);
    var b: std.ArrayList(Lv) = .empty;
    defer b.deinit(gpa);
    try decodeVersion(history, gpa, b_token, true, &b);
    return history.compareFrontiers(gpa, a.items, b.items);
}

/// Parse the single (name, seq) entry of a version token (compaction bases
/// are single-head by rule).
pub fn versionSingleEntry(token: []const u8) error{Corrupt}!struct { name: []const u8, seq: u64 } {
    var cur = token;
    if (!std.mem.startsWith(u8, cur, version_magic)) return error.Corrupt;
    cur = cur[version_magic.len..];
    if (try getUv(&cur) != 1) return error.Corrupt;
    const name = try getBytes(&cur, 4096);
    if (name.len == 0) return error.Corrupt;
    return .{ .name = name, .seq = try getUv(&cur) };
}

/// Batch agent-table helpers (wire encoding).
pub fn tableAdd(gpa: Allocator, table: *std.ArrayList(AgentId), aid: AgentId) Allocator.Error!void {
    for (table.items) |x| if (x == aid) return;
    try table.append(gpa, aid);
}

pub fn tableIndexOf(table: []const AgentId, aid: AgentId) usize {
    for (table, 0..) |x, i| if (x == aid) return i;
    unreachable;
}

test {
    std.testing.refAllDecls(@This());
}
