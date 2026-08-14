//! TextDoc — a collaborative text document: a `Rope` plus the causal event
//! graph that explains it.
//!
//! The consumer contract, in order of daily use:
//! - `insert`/`delete`: local editing. Records causally-stamped unit events
//!   and applies to the rope. No CRDT metadata on the document itself.
//! - `merge`: integrate a batch of encoded remote events. Applies their
//!   transformed effects to the rope and returns the byte-space `[]Edit`
//!   stream — feed it to the same `AnchorSet.shift` / `Anchor.shift` you
//!   already use for local edits; cursors and marks survive remote edits
//!   with zero extra machinery.
//! - `version` / `eventsSince`: sync bookkeeping. "What have I seen" and
//!   "what does the remote lack", the latter already wire-encoded.
//! - `serialize` / `open`: whole-document persistence (the event graph is
//!   the document of record; the rope is its materialization).
//!
//! Error semantics: on OOM mid-`merge` the event graph remains consistent
//! and nothing leaks, but the rope may reflect a prefix of the batch —
//! recover by rebuilding from `serialize` (or discarding the doc). Wire
//! decoding rejects malformed input with `error.Corrupt` and causally
//! incomplete batches with `error.MissingDependency`; both leave the
//! document untouched.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const walker_mod = @import("walker.zig");
const rope_mod = @import("../rope.zig");
const geometry = @import("../geometry.zig");

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
const Lv = causal.Lv;
const TextOp = walker_mod.TextOp;
const Graph = walker_mod.Graph;
const Walker = walker_mod.Walker;
const ScalarEdit = walker_mod.ScalarEdit;
const Rope = rope_mod.Rope;
const Range = geometry.Range;
const Edit = geometry.Edit;

const wire_magic = "stg\x01";

pub const TextDoc = struct {
    rope: Rope = .empty,
    graph: Graph = .empty,
    agent: ?AgentId = null,

    pub const empty: TextDoc = .{};

    pub const MergeError = Allocator.Error || error{ Corrupt, MissingDependency };

    pub fn deinit(self: *TextDoc, gpa: Allocator) void {
        self.rope.deinit(gpa);
        self.graph.deinit(gpa);
        self.* = .{};
    }

    /// Set the local author identity (required before local edits). `name`
    /// must be globally unique among collaborating replicas (it is also the
    /// deterministic tiebreak for concurrent inserts).
    pub fn setAgent(self: *TextDoc, gpa: Allocator, name: []const u8) Allocator.Error!void {
        self.agent = try self.graph.registerAgent(gpa, name);
    }

    /// Read access to the materialized document.
    pub fn text(self: *const TextDoc) *const Rope {
        return &self.rope;
    }

    // ── Local editing ───────────────────────────────────────────────────

    /// Record and apply a local insert. Same contract as `Rope.insert`
    /// (byte offset on a scalar boundary, valid UTF-8).
    pub fn insert(self: *TextDoc, gpa: Allocator, byte_offset: usize, content: []const u8) Allocator.Error!Edit {
        const agent = self.agent.?; // setAgent first
        const edit: Edit = .{ .offset = byte_offset, .removed = 0, .inserted = content.len };
        if (content.len == 0) return edit;
        const scalar_pos = self.rope.offsetToScalar(byte_offset);

        var i: u64 = 0;
        var it = (std.unicode.Utf8View.init(content) catch unreachable).iterator();
        while (it.nextCodepoint()) |ch| : (i += 1) {
            _ = try self.graph.addLocal(gpa, agent, .{ .ins = .{ .pos = scalar_pos + i, .ch = ch } });
        }
        _ = try self.rope.insert(gpa, byte_offset, content);
        return edit;
    }

    /// Record and apply a local delete. Same contract as `Rope.delete`.
    pub fn delete(self: *TextDoc, gpa: Allocator, range: Range) Allocator.Error!Edit {
        const agent = self.agent.?;
        const edit: Edit = .{ .offset = range.start, .removed = range.len(), .inserted = 0 };
        if (range.isEmpty()) return edit;
        const scalar_start = self.rope.offsetToScalar(range.start);
        const scalar_count = self.rope.offsetToScalar(range.end) - scalar_start;

        for (0..scalar_count) |_| {
            // Each unit deletes at the same position: the next scalar slides
            // into place after the previous unit's deletion.
            _ = try self.graph.addLocal(gpa, agent, .{ .del = scalar_start });
        }
        _ = try self.rope.delete(gpa, range);
        return edit;
    }

    // ── Versions & sync ─────────────────────────────────────────────────
    // A version is an opaque, replica-portable token: agent NAMES + seqs
    // ("stv" 0x01, uv count, per entry: uv name_len, name, uv seq). Local
    // AgentId numbering never crosses the wire — it differs per replica.

    const version_magic = "stv\x01";

    /// The current version frontier as an opaque portable token. Exchange it
    /// with peers (`eventsSince`) or persist it; treat the bytes as opaque.
    /// Caller owns.
    pub fn version(self: *const TextDoc, gpa: Allocator) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, version_magic);
        try putUv(gpa, &out, self.graph.frontier.items.len);
        for (self.graph.frontier.items) |lv| {
            const id = self.graph.idOf(lv);
            const name = self.graph.agentName(id.agent);
            try putUv(gpa, &out, name.len);
            try out.appendSlice(gpa, name);
            try putUv(gpa, &out, id.seq);
        }
        return out.toOwnedSlice(gpa);
    }

    /// Wire-encode every event the holder of `remote_version` (an opaque
    /// token from their `version()`) lacks, in causally valid order. Version
    /// entries we don't know are ignored (the remote is ahead of us there;
    /// duplicates are skipped on their side). Caller owns.
    pub fn eventsSince(
        self: *const TextDoc,
        gpa: Allocator,
        remote_version: []const u8,
    ) (Allocator.Error || error{Corrupt})![]u8 {
        var known: std.ArrayList(Lv) = .empty;
        defer known.deinit(gpa);
        var cur = remote_version;
        if (!std.mem.startsWith(u8, cur, version_magic)) return error.Corrupt;
        cur = cur[version_magic.len..];
        const count = try getUv(&cur);
        if (count > 1 << 20) return error.Corrupt;
        for (0..count) |_| {
            const nlen = try getUv(&cur);
            if (nlen == 0 or nlen > 4096 or nlen > cur.len) return error.Corrupt;
            const name = cur[0..nlen];
            cur = cur[nlen..];
            const seq = try getUv(&cur);
            const agent = self.graph.findAgent(name) orelse continue;
            if (self.graph.lvOf(.{ .agent = agent, .seq = seq })) |lv| {
                try known.append(gpa, lv);
            }
        }
        var missing = try self.graph.missingFrom(gpa, known.items);
        defer missing.deinit(gpa);
        return self.encodeEvents(gpa, missing.items);
    }

    /// The whole document as its event graph (the durable form).
    pub fn serialize(self: *const TextDoc, gpa: Allocator) Allocator.Error![]u8 {
        var missing = try self.graph.missingFrom(gpa, &.{});
        defer missing.deinit(gpa);
        return self.encodeEvents(gpa, missing.items);
    }

    /// Materialize a document from `serialize`d (or any complete) bytes.
    pub fn open(gpa: Allocator, bytes: []const u8) MergeError!TextDoc {
        var doc: TextDoc = .empty;
        errdefer doc.deinit(gpa);
        const edits = try doc.merge(gpa, bytes);
        gpa.free(edits);
        return doc;
    }

    // ── Merge ───────────────────────────────────────────────────────────

    /// Integrate encoded remote events: updates the graph, applies the
    /// transformed effects to the rope, and returns the byte-space edit
    /// stream (caller owns; shift your anchors through it in order).
    /// Duplicate events are skipped; causally incomplete batches are
    /// rejected whole with `error.MissingDependency`.
    pub fn merge(self: *TextDoc, gpa: Allocator, bytes: []const u8) MergeError![]Edit {
        var dec = try Decoder.init(gpa, self, bytes);
        defer dec.deinit(gpa);

        // Validate the whole batch before mutating the graph.
        try dec.validate(self);
        const first_new: Lv = @intCast(self.graph.eventCount());
        var any_new = false;
        for (dec.events.items) |ev| {
            if (self.graph.lvOf(ev.id) != null) continue; // duplicate
            var parent_lvs: std.ArrayList(Lv) = .empty;
            defer parent_lvs.deinit(gpa);
            for (dec.parentsOf(ev)) |pid| {
                try parent_lvs.append(gpa, self.graph.lvOf(pid).?);
            }
            _ = try self.graph.add(gpa, ev.id, parent_lvs.items, ev.op);
            any_new = true;
        }
        if (!any_new) return try gpa.alloc(Edit, 0);

        // Replay from genesis; events before first_new are silent.
        var w = Walker.init(&self.graph);
        defer w.deinit(gpa);
        var scalar_edits: std.ArrayList(ScalarEdit) = .empty;
        defer scalar_edits.deinit(gpa);
        try w.replayAll(gpa, first_new, &scalar_edits);

        // Apply to the rope, converting scalar → byte space, coalescing runs.
        var edits: std.ArrayList(Edit) = .empty;
        errdefer edits.deinit(gpa);
        var buf: [4]u8 = undefined;
        for (scalar_edits.items) |se| {
            switch (se) {
                .ins => |ins| {
                    const off = self.rope.scalarToOffset(ins.pos);
                    const len = std.unicode.utf8Encode(ins.ch, &buf) catch unreachable;
                    _ = try self.rope.insert(gpa, off, buf[0..len]);
                    if (edits.items.len > 0) {
                        const last = &edits.items[edits.items.len - 1];
                        if (last.removed == 0 and off == last.offset + last.inserted) {
                            last.inserted += len;
                            continue;
                        }
                    }
                    try edits.append(gpa, .{ .offset = off, .removed = 0, .inserted = len });
                },
                .del => |pos| {
                    const start = self.rope.scalarToOffset(pos);
                    const end = self.rope.scalarToOffset(pos + 1);
                    _ = try self.rope.delete(gpa, .{ .start = start, .end = end });
                    if (edits.items.len > 0) {
                        const last = &edits.items[edits.items.len - 1];
                        if (last.inserted == 0 and start == last.offset) {
                            last.removed += end - start;
                            continue;
                        }
                    }
                    try edits.append(gpa, .{ .offset = start, .removed = end - start, .inserted = 0 });
                },
            }
        }
        return edits.toOwnedSlice(gpa);
    }

    // ── Wire format ─────────────────────────────────────────────────────
    // "stg" 0x01
    //   uv agent_count, then per agent: uv name_len, name bytes
    //   uv event_count, then per event:
    //     uv agent_idx (batch table), uv seq
    //     uv parent_count, per parent: uv agent_idx, uv seq
    //     u8 tag (0 = ins, 1 = del), uv pos, [uv ch if ins]
    // All integers are unsigned LEB128. The format is versioned by the
    // magic's trailing byte.

    fn encodeEvents(self: *const TextDoc, gpa: Allocator, lvs: []const Lv) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, wire_magic);

        // Batch agent table: every agent appearing as author or parent.
        var table: std.ArrayList(AgentId) = .empty;
        defer table.deinit(gpa);
        for (lvs) |lv| {
            try tableAdd(gpa, &table, self.graph.idOf(lv).agent);
            for (self.graph.parentsOf(lv)) |p| {
                try tableAdd(gpa, &table, self.graph.idOf(p).agent);
            }
        }
        try putUv(gpa, &out, table.items.len);
        for (table.items) |aid| {
            const name = self.graph.agentName(aid);
            try putUv(gpa, &out, name.len);
            try out.appendSlice(gpa, name);
        }

        try putUv(gpa, &out, lvs.len);
        for (lvs) |lv| {
            const id = self.graph.idOf(lv);
            try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
            try putUv(gpa, &out, id.seq);
            const parents = self.graph.parentsOf(lv);
            try putUv(gpa, &out, parents.len);
            for (parents) |p| {
                const pid = self.graph.idOf(p);
                try putUv(gpa, &out, tableIndexOf(table.items, pid.agent));
                try putUv(gpa, &out, pid.seq);
            }
            switch (self.graph.opOf(lv)) {
                .ins => |ins| {
                    try out.append(gpa, 0);
                    try putUv(gpa, &out, ins.pos);
                    try putUv(gpa, &out, ins.ch);
                },
                .del => |pos| {
                    try out.append(gpa, 1);
                    try putUv(gpa, &out, pos);
                },
            }
        }
        return out.toOwnedSlice(gpa);
    }

    fn tableAdd(gpa: Allocator, table: *std.ArrayList(AgentId), aid: AgentId) Allocator.Error!void {
        for (table.items) |x| if (x == aid) return;
        try table.append(gpa, aid);
    }

    fn tableIndexOf(table: []const AgentId, aid: AgentId) usize {
        for (table, 0..) |x, i| if (x == aid) return i;
        unreachable;
    }

    const DecodedEvent = struct {
        id: EventId,
        parents_start: u32,
        parents_len: u32,
        op: TextOp,
    };

    const Decoder = struct {
        events: std.ArrayList(DecodedEvent) = .empty,
        parents_pool: std.ArrayList(EventId) = .empty,

        fn parentsOf(self: *const Decoder, ev: DecodedEvent) []const EventId {
            return self.parents_pool.items[ev.parents_start..][0..ev.parents_len];
        }

        fn deinit(self: *Decoder, gpa: Allocator) void {
            self.events.deinit(gpa);
            self.parents_pool.deinit(gpa);
        }

        fn init(gpa: Allocator, doc: *TextDoc, bytes: []const u8) MergeError!Decoder {
            var self: Decoder = .{};
            errdefer self.deinit(gpa);
            var cur: []const u8 = bytes;

            if (!std.mem.startsWith(u8, cur, wire_magic)) return error.Corrupt;
            cur = cur[wire_magic.len..];

            const agent_count = try getUv(&cur);
            if (agent_count > 1 << 20) return error.Corrupt;
            var agents: std.ArrayList(AgentId) = .empty;
            defer agents.deinit(gpa);
            for (0..agent_count) |_| {
                const nlen = try getUv(&cur);
                if (nlen == 0 or nlen > 4096 or nlen > cur.len) return error.Corrupt;
                const name = cur[0..nlen];
                cur = cur[nlen..];
                try agents.append(gpa, try doc.graph.registerAgent(gpa, name));
            }

            const event_count = try getUv(&cur);
            for (0..event_count) |_| {
                const aidx = try getUv(&cur);
                if (aidx >= agents.items.len) return error.Corrupt;
                const seq = try getUv(&cur);
                const pcount = try getUv(&cur);
                if (pcount > 1 << 16) return error.Corrupt;
                const pstart: u32 = @intCast(self.parents_pool.items.len);
                for (0..pcount) |_| {
                    const paidx = try getUv(&cur);
                    if (paidx >= agents.items.len) return error.Corrupt;
                    const pseq = try getUv(&cur);
                    try self.parents_pool.append(gpa, .{ .agent = agents.items[paidx], .seq = pseq });
                }
                if (cur.len == 0) return error.Corrupt;
                const tag = cur[0];
                cur = cur[1..];
                const op: TextOp = switch (tag) {
                    0 => blk: {
                        const pos = try getUv(&cur);
                        const ch = try getUv(&cur);
                        if (ch > std.math.maxInt(u21) or !std.unicode.utf8ValidCodepoint(@intCast(ch)))
                            return error.Corrupt;
                        break :blk .{ .ins = .{ .pos = pos, .ch = @intCast(ch) } };
                    },
                    1 => .{ .del = try getUv(&cur) },
                    else => return error.Corrupt,
                };
                try self.events.append(gpa, .{
                    .id = .{ .agent = agents.items[aidx], .seq = seq },
                    .parents_start = pstart,
                    .parents_len = @intCast(pcount),
                    .op = op,
                });
            }
            return self;
        }

        /// Whole-batch causal validation before any graph mutation: per-agent
        /// contiguity and parent resolvability (existing or earlier-in-batch).
        fn validate(self: *const Decoder, doc: *const TextDoc) error{MissingDependency}!void {
            for (self.events.items, 0..) |ev, i| {
                if (doc.graph.lvOf(ev.id) != null) continue; // duplicate, ignored
                // Non-duplicate ⇒ seq ≥ nextSeq: valid iff it is exactly next,
                // or its per-agent predecessor appears earlier in this batch.
                const next = doc.graph.nextSeq(ev.id.agent);
                const contiguous = ev.id.seq == next or
                    (ev.id.seq > 0 and seenEarlier(self, i, .{ .agent = ev.id.agent, .seq = ev.id.seq - 1 }));
                if (!contiguous) return error.MissingDependency;
                for (self.parentsOf(ev)) |pid| {
                    if (doc.graph.lvOf(pid) == null and !seenEarlier(self, i, pid)) {
                        return error.MissingDependency;
                    }
                }
            }
        }

        fn seenEarlier(self: *const Decoder, before: usize, id: EventId) bool {
            for (self.events.items[0..before]) |ev| {
                if (ev.id.agent == id.agent and ev.id.seq == id.seq) return true;
            }
            return false;
        }
    };
};

// LEB128 helpers over byte slices.
fn putUv(gpa: Allocator, out: *std.ArrayList(u8), value: u64) Allocator.Error!void {
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

fn getUv(cur: *[]const u8) error{Corrupt}!u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    for (0..10) |_| {
        if (cur.len == 0) return error.Corrupt;
        const byte = cur.*[0];
        cur.* = cur.*[1..];
        result |= @as(u64, byte & 0x7f) << shift;
        if (byte & 0x80 == 0) return result;
        if (shift >= 56) return error.Corrupt;
        shift += 7;
    }
    return error.Corrupt;
}

test {
    std.testing.refAllDecls(@This());
}
