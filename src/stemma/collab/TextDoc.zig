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
//! - `version` / `eventsSince`: sync bookkeeping, both ends opaque and
//!   wire-ready.
//! - `serialize` / `open`: whole-document persistence (the event graph is
//!   the document of record; the rope is its materialization).
//! - `materializeAt`: time travel to any known version.
//! - `anchorAt` / `resolveAnchors`: portable identity positions (presence,
//!   remote cursors) that survive concurrent edits.
//! - `compact`: collapse all-peers-stable history into a frozen base
//!   snapshot, bounding graph growth for long-lived documents.
//!
//! ## Compaction model
//! `compact(stable)` requires a *linearization point*: `stable` must be a
//! single-head version and every retained event must causally descend from
//! it (checked; `error.NotCompactable` otherwise). Compacted history
//! becomes an opaque base text; per-agent sequence watermarks keep
//! duplicate detection exact. Sync after compaction requires the peer to
//! share the same base (or bootstrap from scratch): batches referencing
//! compacted *interior* history are rejected with
//! `error.MissingDependency`. Identity anchors cannot target compacted
//! characters (`error.Compacted`).
//!
//! ## Error semantics
//! Rejected batches (`Corrupt`, `MissingDependency`) leave the document
//! untouched (graph-phase rollback). On OOM mid-`merge` the event graph
//! remains consistent and nothing leaks, but the rope may reflect a prefix
//! of the batch — recover by rebuilding from `serialize`. Equivocation (a
//! peer emitting two different events with the same id) is undetectable at
//! this layer, as in any unauthenticated CRDT — authenticate peers in the
//! transport if your threat model needs it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const core = @import("core.zig");
const seq_walker = @import("SeqWalker.zig");
// TextDoc's single sequence spans the WHOLE document and every event in
// its graph is a sequence op — a dense array indexed by `lv` is a
// perfect-fit O(1) slot per event (see `SeqWalker.Storage`).
const SeqWalker = seq_walker.SeqWalker(.dense);
const rope_mod = @import("../rope.zig");
const geometry = @import("../geometry.zig");
const wire = @import("wire.zig");
const putUv = wire.putUv;
const getUv = wire.getUv;

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
const Lv = causal.Lv;
const Rope = rope_mod.Rope;
const Range = geometry.Range;
const Edit = geometry.Edit;

/// Text operation payload, in **scalar** (Unicode codepoint) coordinates —
/// portable across replicas regardless of their byte encodings.
pub const TextOp = union(enum) {
    /// Insert scalar `ch` at scalar position `pos` (as seen by the author).
    ins: struct { pos: u64, ch: u21 },
    /// Delete the scalar at position `pos` (as seen by the author).
    del: u64,
};

pub const Graph = causal.EventGraph(TextOp);

/// A transformed operation in scalar space, valid against the document
/// state produced by all previously emitted edits.
const ScalarEdit = union(enum) {
    ins: struct { pos: u64, ch: u21 },
    del: u64,
};

/// Sentinel `lv` for compacted base placeholder items.
const base_lv: Lv = std.math.maxInt(Lv);

const wire_magic_v1 = "stg\x01";
const wire_magic_v2 = "stg\x02";
const wire_magic_v3 = "stg\x03";
const version_magic = core.version_magic;

/// Batch encoding produced by `eventsSince`/`serialize`. Decoding always
/// accepts every version.
pub const WireFormat = enum {
    /// Run-RLE frames ("stg" 0x03): a monotone typing or deletion burst
    /// costs one frame instead of one event per character. The default.
    rle,
    /// Per-unit events ("stg" 0x01 / 0x02) for pre-RLE decoders.
    unit,
};

/// Replay of the whole history against one shared `Sequence`: the
/// eg-walker prepare/effect discipline (see `SeqWalker.zig` for the
/// retreat/advance/apply mechanics this drives, `Sequence.zig` for the
/// ordering, `causal.zig` for the graph). Events before `emit_from`
/// replay silently; newer (untrusted) events emit transformed scalar
/// edits and get error-checked positions instead of asserts.
///
/// Holds no reference to a `Graph` — every method takes `history`
/// explicitly — so a `Replay` can be parked as a long-lived cache (see
/// `TextDoc.merge_walk`) across calls without risking a dangling pointer
/// if the owning `TextDoc` is copied or `history` is rebuilt by
/// `compact`. `walked` is the exclusive upper bound of what's already
/// folded into `sw`: a resumed `replayAll` only walks `[walked, target)`,
/// not from genesis every time.
const Replay = struct {
    sw: SeqWalker = .empty,
    prep_frontier: std.ArrayList(Lv) = .empty,
    walked: Lv = 0,

    const empty: Replay = .{};

    const Names = struct {
        g: *const Graph,
        pub fn agentNameOf(self: @This(), lv: Lv) []const u8 {
            // Base placeholders sort as the empty name: causally before
            // everything live; the rule only needs cross-replica determinism.
            if (lv == base_lv) return "";
            return self.g.agentName(self.g.idOf(lv).agent);
        }
    };

    fn deinit(self: *Replay, gpa: Allocator) void {
        self.sw.deinit(gpa);
        self.prep_frontier.deinit(gpa);
    }

    /// Discard all cached state (a resumed `replayAll` starts from
    /// genesis again). Called wherever the `lv` space it was built
    /// against stops being valid: a rejected merge (partial walk against
    /// events `rollbackHistory` is about to erase) or `compact`
    /// (renumbers everything).
    fn reset(self: *Replay, gpa: Allocator) void {
        self.deinit(gpa);
        self.* = .empty;
    }

    /// Must precede any live item; only valid when the cache is fresh
    /// (`walked == 0`), since the placeholders must sit first in `sw`.
    fn initBase(self: *Replay, gpa: Allocator, count: usize) Allocator.Error!void {
        // A `TextDoc` base only ever comes from `compact`, never from a content
        // load, so its scalars stay unaddressable (`SeqWalker.base_id`).
        try self.sw.initBase(gpa, count, base_lv, null);
    }

    const ReplayError = Allocator.Error || error{Corrupt};

    /// Walk `[self.walked, target)` against `history`, advancing the
    /// cache to `target`. `emit_from` (>= self.walked) gates which of
    /// those events surface a `ScalarEdit` — events before it (already
    /// applied to the document some other way, e.g. a local edit that
    /// touched the rope directly, or a prior `replayAll` call) replay
    /// for bookkeeping only.
    fn replayAll(
        self: *Replay,
        gpa: Allocator,
        history: *const Graph,
        target: Lv,
        emit_from: Lv,
        include: ?[]const bool,
        out: *std.ArrayList(ScalarEdit),
    ) ReplayError!void {
        var lv = self.walked;
        while (lv < target) : (lv += 1) {
            if (include) |inc| if (!inc[lv]) continue;
            try seq_walker.movePrepareTo(gpa, history, &self.prep_frontier, history.parentsOf(lv), self);
            const emit = lv >= emit_from;
            switch (history.opOf(lv)) {
                .ins => |ins| {
                    const res = try self.sw.applyInsert(gpa, Names{ .g = history }, lv, ins.pos, emit);
                    if (emit) try out.append(gpa, .{ .ins = .{ .pos = res.effect_pos, .ch = ins.ch } });
                },
                .del => |pos| {
                    const res = try self.sw.applyDelete(gpa, lv, pos, emit);
                    if (emit) {
                        if (res.effect_pos) |p| try out.append(gpa, .{ .del = p });
                    }
                },
            }
            self.prep_frontier.clearRetainingCapacity();
            try self.prep_frontier.append(gpa, lv);
        }
        self.walked = target;
    }

    /// `SeqWalker.movePrepareTo`'s per-event callback.
    pub fn toggle(self: *Replay, history: *const Graph, lv: Lv, on: bool) void {
        switch (history.opOf(lv)) {
            .ins => self.sw.toggleInsert(lv, on),
            .del => self.sw.toggleDelete(lv, on),
        }
    }
};

const TextDoc = @This();
rope: Rope = .empty,
history: Graph = .empty,
agent: ?AgentId = null,

/// Frozen compacted pre-history (UTF-8), or empty. For a partially
/// realized base (`openPartial`) the unfetched spans are zero-filled
/// until `realizeBase` supplies them; `holes` says which.
base_bytes: []u8 = &.{},
base_scalars: usize = 0,
/// The opaque version token identifying the base; docs can only sync
/// when their bases are identical (or one side bootstraps).
base_version: []u8 = &.{},
/// The single stable head the base was compacted at.
base_head: ?EventId = null,
/// Unrealized spans of the base, ascending by `cur_offset`. Empty =
/// fully realized (the only state most documents ever see).
holes: std.ArrayList(BaseHole) = .empty,

/// Persistent eg-walker cache for the live head: `merge` resumes it
/// instead of replaying `history` from genesis every call, so cost
/// tracks events since the last merge, not document lifetime. Reset
/// (not incrementally repaired) whenever the `lv` space it was built
/// against stops being valid: a rejected merge, or `compact` (which
/// renumbers every retained event).
merge_walk: Replay = .empty,

/// An unrealized span of the compacted base. `base_offset` is the fetch
/// key (byte offset in the pristine base — immutable); `cur_offset` is
/// where the span currently sits in the rope (shifts under edits; hole
/// interiors themselves are never edited).
pub const BaseHole = struct {
    base_offset: usize,
    cur_offset: usize,
    bytes: usize,
    scalars: usize,
};

pub const empty: TextDoc = .{};

pub const MergeError = Allocator.Error || error{ Corrupt, MissingDependency, Unrealized };
pub const CompactError = MergeError || error{NotCompactable};
pub const AnchorError = seq_walker.AnchorError;

pub fn deinit(self: *TextDoc, gpa: Allocator) void {
    self.rope.deinit(gpa);
    self.history.deinit(gpa);
    gpa.free(self.base_bytes);
    gpa.free(self.base_version);
    self.holes.deinit(gpa);
    self.merge_walk.deinit(gpa);
    self.* = .{};
}

/// Set the local author identity (required before local edits). `name`
/// must be globally unique among collaborating replicas (it is also the
/// deterministic tiebreak for concurrent inserts).
pub fn setAgent(self: *TextDoc, gpa: Allocator, name: []const u8) Allocator.Error!void {
    self.agent = try self.history.registerAgent(gpa, name);
}

/// Raw event count since the last compaction (or since genesis,
/// uncompacted) — replay cost scales with this. Delegates to `history`
/// so callers never need to reach through `TextDoc`'s internal `Graph`
/// field.
pub fn eventCount(self: *const TextDoc) usize {
    return self.history.eventCount();
}

/// Read access to the materialized document.
pub fn text(self: *const TextDoc) *const Rope {
    return &self.rope;
}

// ── Local editing ───────────────────────────────────────────────────

/// Record and apply a local insert. Same contract as `Rope.insert`
/// (byte offset on a scalar boundary, valid UTF-8). Precondition: the
/// offset is not interior to an unrealized base span (deterministic
/// panic, all build modes — `realizeBase` first).
pub fn insert(self: *TextDoc, gpa: Allocator, byte_offset: usize, content: []const u8) Allocator.Error!Edit {
    const agent = self.agent.?; // setAgent first
    const edit: Edit = .{ .offset = byte_offset, .removed = 0, .inserted = content.len };
    if (content.len == 0) return edit;
    self.assertOutsideHoles(byte_offset, byte_offset);
    const scalar_pos = self.rope.offsetToScalar(byte_offset) + self.holeScalarsThrough(byte_offset);

    var i: u64 = 0;
    var it = (std.unicode.Utf8View.init(content) catch unreachable).iterator();
    while (it.nextCodepoint()) |ch| : (i += 1) {
        _ = try self.history.addLocal(gpa, agent, .{ .ins = .{ .pos = scalar_pos + i, .ch = ch } });
    }
    _ = try self.rope.insert(gpa, byte_offset, content);
    self.shiftHoles(byte_offset, content.len, 0);
    return edit;
}

/// Record and apply a local delete. Same contract as `Rope.delete`.
/// Precondition: the range does not intersect an unrealized base span
/// (deterministic panic, all build modes — `realizeBase` first).
pub fn delete(self: *TextDoc, gpa: Allocator, range: Range) Allocator.Error!Edit {
    const agent = self.agent.?;
    const edit: Edit = .{ .offset = range.start, .removed = range.len(), .inserted = 0 };
    if (range.isEmpty()) return edit;
    self.assertOutsideHoles(range.start, range.end);
    const scalar_start = self.rope.offsetToScalar(range.start) + self.holeScalarsThrough(range.start);
    const scalar_count = self.rope.offsetToScalar(range.end) - self.rope.offsetToScalar(range.start);

    for (0..scalar_count) |_| {
        // Each unit deletes at the same position: the next scalar slides
        // into place after the previous unit's deletion.
        _ = try self.history.addLocal(gpa, agent, .{ .del = scalar_start });
    }
    _ = try self.rope.delete(gpa, range);
    self.shiftHoles(range.start, 0, range.len());
    return edit;
}

/// Panic if `[start, end]` touches the interior of an unrealized base
/// span (or covers one). Single choke point, all build modes — mirrors
/// `Rope`'s hole-content panic.
fn assertOutsideHoles(self: *const TextDoc, start: usize, end: usize) void {
    for (self.holes.items) |h| {
        if (end > h.cur_offset and start < h.cur_offset + h.bytes) {
            @panic("stemma.TextDoc: edit touches an unrealized base span — realizeBase() first");
        }
    }
}

/// Scalars of unrealized base spans at or before `byte_offset` (spans
/// ending exactly at the offset count; spans starting there do not).
fn holeScalarsThrough(self: *const TextDoc, byte_offset: usize) u64 {
    var acc: u64 = 0;
    for (self.holes.items) |h| {
        if (h.cur_offset + h.bytes <= byte_offset) acc += h.scalars;
    }
    return acc;
}

/// Shift hole positions through a byte edit at `at`. Holes never
/// intersect edits (checked by callers), so a whole-span shift is exact.
fn shiftHoles(self: *TextDoc, at: usize, inserted: usize, removed: usize) void {
    for (self.holes.items) |*h| {
        if (h.cur_offset >= at) h.cur_offset = h.cur_offset + inserted - removed;
    }
}

// ── Versions & sync ─────────────────────────────────────────────────
// A version is an opaque, replica-portable token: agent NAMES + seqs
// ("stv" 0x01, uv count, per entry: uv name_len, name, uv seq). Local
// AgentId numbering never crosses the wire — it differs per replica.

/// The current version frontier as an opaque portable token. Exchange it
/// with peers (`eventsSince`) or persist it; treat the bytes as opaque.
/// Caller owns.
pub fn version(self: *const TextDoc, gpa: Allocator) Allocator.Error![]u8 {
    return core.encodeVersion(gpa, &self.history);
}

fn decodeVersion(
    self: *const TextDoc,
    gpa: Allocator,
    token: []const u8,
    strict: bool,
    out: *std.ArrayList(Lv),
) MergeError!void {
    return core.decodeVersion(&self.history, gpa, token, strict, out);
}

pub const VersionOrder = causal.VersionOrder;

/// Causal relation between two version tokens (from `a`'s perspective:
/// `.ancestor` = `a` is strictly earlier). Every entry of both tokens
/// must be stored by this replica (`error.MissingDependency` otherwise —
/// e.g. tokens from ahead-of-us peers or from below the compaction
/// horizon). Typical use: "is my saved version stale?", "did these two
/// diverge?".
pub fn compareVersions(
    self: *const TextDoc,
    gpa: Allocator,
    a_token: []const u8,
    b_token: []const u8,
) MergeError!VersionOrder {
    return core.compareVersions(gpa, &self.history, a_token, b_token);
}

const versionSingleEntry = core.versionSingleEntry;

/// Wire-encode every event the holder of `remote_version` (an opaque
/// token from their `version()`) lacks, in causally valid order. Version
/// entries we don't know are ignored (the remote is ahead of us there;
/// duplicates are skipped on their side). Caller owns.
pub fn eventsSince(
    self: *const TextDoc,
    gpa: Allocator,
    remote_version: []const u8,
) (Allocator.Error || error{Corrupt})![]u8 {
    return self.eventsSinceFormat(gpa, remote_version, .rle) catch |e| switch (e) {
        error.Unrealized => unreachable, // rle syncs never need base bytes
        else => |err| return err,
    };
}

/// `eventsSince` with an explicit batch encoding — serve `.unit` to
/// peers that predate run-RLE. A partially realized document cannot
/// emit `.unit` (that format always carries base bytes):
/// `error.Unrealized`.
pub fn eventsSinceFormat(
    self: *const TextDoc,
    gpa: Allocator,
    remote_version: []const u8,
    format: WireFormat,
) (Allocator.Error || error{ Corrupt, Unrealized })![]u8 {
    if (format == .unit and self.holes.items.len != 0) return error.Unrealized;
    var known: std.ArrayList(Lv) = .empty;
    defer known.deinit(gpa);
    self.decodeVersion(gpa, remote_version, false, &known) catch |e| switch (e) {
        error.MissingDependency => unreachable, // lenient mode
        else => |err| return err,
    };
    var missing = try self.history.missingFrom(gpa, known.items);
    defer missing.deinit(gpa);
    return self.encodeEvents(gpa, missing.items, format);
}

/// Wire-encode the events in `to_version`'s causal past that the holder
/// of `from_version` lacks — `eventsSince` bounded above by `to` instead
/// of the current head. The incremental-sync primitive for consumers
/// tracking a past state (persistence deltas, a saved-file mirror).
/// Every entry of `to_version` must be stored here
/// (`error.MissingDependency`); unknown `from` entries are ignored.
pub fn eventsBetween(
    self: *const TextDoc,
    gpa: Allocator,
    from_version: []const u8,
    to_version: []const u8,
) MergeError![]u8 {
    var known: std.ArrayList(Lv) = .empty;
    defer known.deinit(gpa);
    self.decodeVersion(gpa, from_version, false, &known) catch |e| switch (e) {
        error.MissingDependency => unreachable, // lenient mode
        else => |err| return err,
    };
    var to_heads: std.ArrayList(Lv) = .empty;
    defer to_heads.deinit(gpa);
    try self.decodeVersion(gpa, to_version, true, &to_heads);
    var d = try self.history.diff(gpa, to_heads.items, known.items);
    defer d.deinit(gpa);
    return self.encodeEvents(gpa, d.a_only.items, .rle);
}

/// The whole document as its event graph (plus the base snapshot when
/// compacted) — the durable form. A partially realized base cannot be
/// persisted (`error.Unrealized`): realize it first.
pub fn serialize(self: *const TextDoc, gpa: Allocator) (Allocator.Error || error{Unrealized})![]u8 {
    if (self.holes.items.len != 0) return error.Unrealized;
    var missing = try self.history.missingFrom(gpa, &.{});
    defer missing.deinit(gpa);
    return self.encodeEvents(gpa, missing.items, .rle);
}

/// Materialize a document from `serialize`d (or any complete) bytes.
pub fn open(gpa: Allocator, bytes: []const u8) MergeError!TextDoc {
    var doc: TextDoc = .empty;
    errdefer doc.deinit(gpa);
    const edits = try doc.merge(gpa, bytes);
    gpa.free(edits);
    return doc;
}

// ── Partial checkout ────────────────────────────────────────────────
// A document whose compacted base is only partially fetched: realized
// chunks carry content, holes carry (bytes, scalars) metadata — exactly
// what a host's chunk table supplies. The event graph is complete from
// the start (versions, sync, and merges all work); only *content* is
// lazy. Hole interiors are protected regions: local edits there are a
// precondition violation, remote merges into them roll back whole with
// `error.Unrealized` (realize, then merge again). `realizeBase` is not
// an edit — offsets, anchors, and versions are unaffected.

/// One span of a partial base, in pristine-base order.
pub const BaseChunk = union(enum) {
    /// Fetched content (UTF-8, copied).
    content: []const u8,
    /// Unfetched span of known size.
    hole: struct { bytes: usize, scalars: usize },
};

/// Per-agent compaction watermark — what a host sends alongside its
/// `base_version` and chunk table so a partial replica can join.
pub const AgentWatermark = struct { name: []const u8, seq_base: u64 };

/// This document's watermarks for serving `openPartial` peers. Names
/// borrow from the document (valid until deinit); caller frees the
/// slice only. CAVEAT: `compact` rebuilds `self.history` onto a brand
/// new `Graph` (a fresh `names` arena via `registerAgent`, the old one
/// freed by the old graph's `deinit`) — a `name` borrowed here and held
/// across an intervening `compact()` call dangles, it does not merely go
/// stale. Safe as used today (a single-threaded serve call reads,
/// encodes, and frees the returned slice within one synchronous call,
/// never interleaved with a `compact()`); a caller that caches this
/// across calls must re-fetch after every `compact`.
pub fn agentWatermarks(self: *const TextDoc, gpa: Allocator) Allocator.Error![]AgentWatermark {
    const out = try gpa.alloc(AgentWatermark, self.history.agents.items.len);
    for (out, self.history.agents.items, 0..) |*w, a, i| {
        w.* = .{
            .name = self.history.agentName(@enumFromInt(i)),
            .seq_base = a.seq_base,
        };
    }
    return out;
}

/// Bootstrap a partially realized replica of a compacted document:
/// `base_version` (a single-head token) plus `agents` watermarks come
/// from the host (`agentWatermarks`), `chunks` describe the base in
/// order — fetched content or holes. Sync works immediately; content
/// inside holes waits for `realizeBase`.
pub fn openPartial(
    gpa: Allocator,
    base_version: []const u8,
    agents: []const AgentWatermark,
    chunks: []const BaseChunk,
) (Allocator.Error || error{Corrupt})!TextDoc {
    var doc: TextDoc = .empty;
    errdefer doc.deinit(gpa);

    for (agents) |w| {
        const aid = try doc.history.registerAgent(gpa, w.name);
        doc.history.agents.items[@intFromEnum(aid)].seq_base = w.seq_base;
    }
    if (doc.history.agents.items.len != agents.len) return error.Corrupt; // duplicate names
    const head = try versionSingleEntry(base_version);
    const head_agent = doc.history.findAgent(head.name) orelse return error.Corrupt;
    if (head.seq >= doc.history.agents.items[@intFromEnum(head_agent)].seq_base) {
        return error.Corrupt; // the base head must itself be compacted
    }
    doc.base_head = .{ .agent = head_agent, .seq = head.seq };
    doc.base_version = try gpa.dupe(u8, base_version);

    var total_bytes: usize = 0;
    var total_scalars: usize = 0;
    for (chunks) |c| switch (c) {
        .content => |bytes| {
            const scalars = std.unicode.utf8CountCodepoints(bytes) catch return error.Corrupt;
            total_bytes += bytes.len;
            total_scalars += scalars;
        },
        .hole => |h| {
            if (h.bytes == 0 or h.scalars == 0 or h.scalars > h.bytes) return error.Corrupt;
            total_bytes += h.bytes;
            total_scalars += h.scalars;
        },
    };
    doc.base_scalars = total_scalars;
    const base = try gpa.alloc(u8, total_bytes);
    doc.base_bytes = base;
    @memset(base, 0);

    var offset: usize = 0;
    for (chunks) |c| switch (c) {
        .content => |bytes| {
            if (bytes.len > 0) {
                @memcpy(base[offset..][0..bytes.len], bytes);
                var piece = try Rope.fromSlice(gpa, bytes);
                defer piece.deinit(gpa);
                try doc.rope.append(gpa, &piece);
            }
            offset += bytes.len;
        },
        .hole => |h| {
            try doc.holes.append(gpa, .{
                .base_offset = offset,
                .cur_offset = offset,
                .bytes = h.bytes,
                .scalars = h.scalars,
            });
            var piece = try Rope.fromUnrealized(gpa, h.bytes);
            defer piece.deinit(gpa);
            try doc.rope.append(gpa, &piece);
            offset += h.bytes;
        },
    };
    return doc;
}

/// Open a document whose entire initial content is a compacted base —
/// the bulk-load path: O(content) with ZERO events, where recording a
/// load as insert events costs one event per scalar. The synthetic
/// base head is derived from the content hash, so replicas that load
/// identical bytes independently share a history root (they can sync,
/// and divergent checkouts of the same file adopt the same root by
/// construction), while different contents can never be confused for
/// the same base. Identity anchors into the base resolve as
/// `error.Compacted`, like any compacted content.
pub fn openFromContent(gpa: Allocator, content: []const u8) (Allocator.Error || error{Corrupt})!TextDoc {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(content, &digest, .{});
    var name_buf: [5 + 32]u8 = undefined;
    @memcpy(name_buf[0..5], "base-");
    for (digest[0..16], 0..) |b, i| {
        _ = std.fmt.bufPrint(name_buf[5 + i * 2 ..][0..2], "{x:0>2}", .{b}) catch unreachable;
    }
    const name = name_buf[0..];

    var token: std.ArrayList(u8) = .empty;
    defer token.deinit(gpa);
    try token.appendSlice(gpa, version_magic);
    try putUv(gpa, &token, 1);
    try putUv(gpa, &token, name.len);
    try token.appendSlice(gpa, name);
    try putUv(gpa, &token, 0); // the synthetic head: (name, seq 0)

    return openPartial(gpa, token.items, &.{
        .{ .name = name, .seq_base = 1 },
    }, &.{
        .{ .content = content },
    });
}

/// Whether every byte of the base is realized. Always true for
/// documents that never went through `openPartial`.
pub fn baseRealized(self: *const TextDoc) bool {
    return self.holes.items.len == 0;
}

/// The fetch list: unrealized base spans, `base_offset` being the byte
/// range key in the pristine base. Borrows from the document.
pub fn unrealizedBase(self: *const TextDoc) []const BaseHole {
    return self.holes.items;
}

/// Supply the pristine content of one unrealized span (whole-span; the
/// span is identified by its `base_offset`). Verified against the
/// recorded byte and scalar counts (`error.Corrupt` on mismatch —
/// nothing changes). Not an edit: offsets, anchors, versions are all
/// unaffected.
pub fn realizeBase(
    self: *TextDoc,
    gpa: Allocator,
    base_offset: usize,
    content: []const u8,
) (Allocator.Error || error{Corrupt})!void {
    const idx = for (self.holes.items, 0..) |h, i| {
        if (h.base_offset == base_offset) break i;
    } else return error.Corrupt;
    const h = self.holes.items[idx];
    if (content.len != h.bytes) return error.Corrupt;
    if (!std.unicode.utf8ValidateSlice(content)) return error.Corrupt;
    if ((std.unicode.utf8CountCodepoints(content) catch unreachable) != h.scalars) return error.Corrupt;
    try self.rope.realize(gpa, h.cur_offset, content);
    @memcpy(self.base_bytes[h.base_offset..][0..h.bytes], content);
    _ = self.holes.orderedRemove(idx);
}

// ── Time travel ─────────────────────────────────────────────────────

/// Materialize the document as it was at `version` (a token from
/// `version()`, ours or a peer's — every entry must be stored by us).
/// Returns a fresh Rope the caller owns. O(history) replay.
pub fn materializeAt(self: *const TextDoc, gpa: Allocator, version_token: []const u8) MergeError!Rope {
    if (self.holes.items.len != 0) return error.Unrealized; // needs base content
    var heads: std.ArrayList(Lv) = .empty;
    defer heads.deinit(gpa);
    try self.decodeVersion(gpa, version_token, true, &heads);

    const n = self.history.eventCount();
    const include = try gpa.alloc(bool, n);
    defer gpa.free(include);
    @memset(include, false);
    var d = try self.history.diff(gpa, heads.items, &.{});
    defer d.deinit(gpa);
    for (d.a_only.items) |lv| include[lv] = true;

    var w: Replay = .empty;
    defer w.deinit(gpa);
    if (self.base_scalars > 0) try w.initBase(gpa, self.base_scalars);
    var sink: std.ArrayList(ScalarEdit) = .empty;
    defer sink.deinit(gpa);
    // Its own scratch walker (not `merge_walk`): an arbitrary historical
    // version has no relation to how much of the live head is cached.
    w.replayAll(gpa, &self.history, @intCast(n), @intCast(n), include, &sink) catch |e| switch (e) {
        error.Corrupt => unreachable, // trusted local history
        else => |err| return err,
    };
    assert(sink.items.len == 0); // fully silent

    const base_chars = try self.decodeBaseScalars(gpa);
    defer gpa.free(base_chars);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var it = w.sw.s.aliveIterator();
    var buf: [4]u8 = undefined;
    while (it.next()) |alive| {
        const ch = if (alive.lv == base_lv)
            base_chars[alive.arena]
        else
            self.history.opOf(alive.lv).ins.ch;
        const len = std.unicode.utf8Encode(ch, &buf) catch unreachable;
        try bytes.appendSlice(gpa, buf[0..len]);
    }
    return Rope.fromSlice(gpa, bytes.items);
}

fn decodeBaseScalars(self: *const TextDoc, gpa: Allocator) Allocator.Error![]u21 {
    const out = try gpa.alloc(u21, self.base_scalars);
    errdefer gpa.free(out);
    var it = (std.unicode.Utf8View.init(self.base_bytes) catch unreachable).iterator();
    var i: usize = 0;
    while (it.nextCodepoint()) |ch| : (i += 1) out[i] = ch;
    assert(i == self.base_scalars);
    return out;
}

// ── Identity anchors ────────────────────────────────────────────────
// Portable positions that survive *concurrent* edits: an anchor names
// the inserting event of a character (agent name + seq — globally
// stable, never a replica-local id) plus a side. Send them to peers as
// presence/remote-cursor positions; resolve against any replica that
// has seen the event. Resolution is O(history) — batch with
// `resolveAnchors` (one replay for the whole set).
//
// The machinery itself (walking `Replay`'s `Sequence` + `history`'s
// `idOf`/`agentName`/`findAgent`/`lvOf`/`isKnown`/`opOf`) is generic over
// any `EventGraph(Op)` + `SeqWalker` pair — nothing here is TextDoc-
// specific — so it lives once in `SeqWalker.zig` (`anchorAt`/
// `resolveAnchors`, `EventAnchor`, `AnchorSide`) and `ObjectDoc` calls the
// same functions per text object (`objectAnchorAt`/`resolveObjectAnchors`,
// stemma-unification.md §3 step 3, delta 3). What stays here: the
// boundary short-circuit (an anchor at the very start/end needs no
// replay), and byte↔scalar conversion via `self.rope` — both are
// TextDoc's own rope's job, not the shared layer's.

pub const AnchorSide = seq_walker.AnchorSide;
pub const EventAnchor = seq_walker.EventAnchor;

/// Anchors only ever name an INSERT event — `resolveAnchors`'s `ctx`.
const AnchorCtx = struct {
    pub fn isInsert(_: @This(), op: TextOp) bool {
        return op == .ins;
    }
};

/// An identity anchor for the position `byte_offset`. `stickiness`
/// chooses the attachment: `.before` attaches to the character at the
/// offset (anchor rides in front of it), `.after` to the character
/// preceding it. Compacted characters — including a still-unrealized
/// base span (partial checkout, `openPartial`) — cannot be anchored
/// (`error.Compacted`). The returned `agent` slice is gpa-owned.
pub fn anchorAt(
    self: *const TextDoc,
    gpa: Allocator,
    byte_offset: usize,
    stickiness: AnchorSide,
) AnchorError!EventAnchor {
    // Global scalar space (what the `SeqWalker` below indexes into)
    // includes unrealized base-hole scalars, which `self.rope`'s own
    // metric does NOT count (a hole leaf's summary carries zero scalars
    // — see `Rope`'s "Lazy / unrealized content" section) —
    // `holeScalarsThrough` compensates, exactly mirroring `insert`'s own
    // byte→scalar step. A no-op (adds 0) for the overwhelming common
    // case of no active holes — this was a found bug (ported into
    // `ObjectDoc.objectAnchorAt` too, fixed there identically): every
    // OTHER byte↔scalar boundary here (`insert`/`delete`, `merge`'s
    // `insByteOffset`/`delByteRange`) already got hole-aware treatment;
    // the anchor path silently didn't, so an anchor taken at or after a
    // hole resolved against the wrong element (interior to the hole's
    // own placeholder block instead of the real content past it).
    const scalar = self.rope.offsetToScalar(byte_offset) + self.holeScalarsThrough(byte_offset);
    const total = self.rope.scalarLen() + self.holeScalarsThrough(self.rope.byteLen());
    switch (stickiness) {
        .before => if (scalar == total) return .{ .agent = "", .side = .after },
        .after => if (scalar == 0) return .{ .agent = "", .side = .before },
    }
    const target_index = switch (stickiness) {
        .before => scalar,
        .after => scalar - 1,
    };

    var w = try self.silentReplay(gpa);
    defer w.deinit(gpa);
    return seq_walker.anchorAt(gpa, &self.history, &w.sw, target_index, stickiness);
}

/// Resolve identity anchors to current byte offsets. Deleted targets
/// collapse to their deletion point. One O(history) replay amortized
/// over the whole batch.
pub fn resolveAnchors(
    self: *const TextDoc,
    gpa: Allocator,
    anchors: []const EventAnchor,
    out: []usize,
) AnchorError!void {
    assert(anchors.len == out.len);
    var w = try self.silentReplay(gpa);
    defer w.deinit(gpa);
    try seq_walker.resolveAnchors(gpa, &self.history, &w.sw, AnchorCtx{}, anchors, out);
    // The shared layer returns GLOBAL SCALAR positions (base-hole-
    // inclusive, same space `anchorAt` targets above) — map to bytes with
    // the same hole-aware boundary mapping `merge`'s insert path uses
    // (`insByteOffset`): a resolved identity's item can only ever sit
    // before or after a hole's contiguous block, never interior to one
    // (no edit is ever accepted interior to a hole —
    // `assertOutsideHoles`/`checkHoleConflicts` both refuse that), so the
    // insertion-point mapping is exactly the boundary mapping a READ
    // needs too. A no-op (degenerates to plain `scalarToOffset`) when
    // there are no active holes — the other half of `anchorAt`'s found
    // bug: the old direct `self.rope.scalarToOffset(o.*)` silently
    // undercounted by every preceding hole's scalars whenever one existed.
    for (out) |*o| o.* = self.insByteOffset(@intCast(o.*));
}

/// Full silent replay of the whole graph (current state). Its own
/// scratch walker, not `merge_walk`: this takes `*const TextDoc` (called
/// from anchor resolution, which makes no mutation promise to callers),
/// so it can't share the mutable merge cache. Still O(history) per call
/// — a candidate for the same caching `merge` got, left for later.
fn silentReplay(self: *const TextDoc, gpa: Allocator) Allocator.Error!Replay {
    var w: Replay = .empty;
    errdefer w.deinit(gpa);
    if (self.base_scalars > 0) try w.initBase(gpa, self.base_scalars);
    var sink: std.ArrayList(ScalarEdit) = .empty;
    defer sink.deinit(gpa);
    const n: Lv = @intCast(self.history.eventCount());
    w.replayAll(gpa, &self.history, n, n, null, &sink) catch |e| switch (e) {
        error.Corrupt => unreachable, // trusted local history
        else => |err| return err,
    };
    assert(sink.items.len == 0);
    return w;
}

// ── Compaction ──────────────────────────────────────────────────────

/// Collapse all history at-or-before `stable_token` into a frozen base
/// snapshot. Requirements (checked): the token names a single event
/// (a linearization point) we have stored, and every retained event
/// causally descends from it through retained events only. Call this
/// when every collaborating peer has acknowledged `stable_token`; peers
/// that have not can no longer sync with us (they must compact to the
/// same point or re-bootstrap). Identity anchors into compacted content
/// stop resolving (`error.Compacted`).
pub fn compact(self: *TextDoc, gpa: Allocator, stable_token: []const u8) CompactError!void {
    var heads: std.ArrayList(Lv) = .empty;
    defer heads.deinit(gpa);
    try self.decodeVersion(gpa, stable_token, true, &heads);
    if (heads.items.len != 1) return error.NotCompactable;
    const s = heads.items[0];

    const n = self.history.eventCount();
    const in_base = try gpa.alloc(bool, n);
    defer gpa.free(in_base);
    @memset(in_base, false);
    var d = try self.history.diff(gpa, &.{s}, &.{});
    defer d.deinit(gpa);
    for (d.a_only.items) |lv| in_base[lv] = true;

    // Every retained event must sit causally after `s`, reachable only
    // through retained events (or `s` itself).
    for (0..n) |lv| {
        if (in_base[lv]) continue;
        for (self.history.parentsOf(@intCast(lv))) |p| {
            if (in_base[p] and p != s) return error.NotCompactable;
        }
    }

    // Materialize the base BEFORE touching the graph.
    var base_rope = try self.materializeAt(gpa, stable_token);
    defer base_rope.deinit(gpa);
    const new_base_bytes = try base_rope.toOwnedSlice(gpa);
    errdefer gpa.free(new_base_bytes);
    const new_base_scalars = base_rope.scalarLen();
    const new_base_version = try gpa.dupe(u8, stable_token);
    errdefer gpa.free(new_base_version);

    // Rebuild the graph: same agents (same order → same AgentIds, so
    // `self.agent` stays valid), watermarks raised by their share of
    // the base, retained events re-added in causal order. Graph-shape-only
    // (never inspects `TextOp`) — shared with `ObjectDoc.compact` as
    // `causal.compactGraph` (`stemma-unification.md` §3 step 4). The
    // reachability check just above guarantees every retained event's only
    // `in_base` parent (if any) is `s` itself, so `compactGraph`'s blanket
    // "drop any `in_base` parent" rule specializes, for TextDoc, to exactly
    // the old "skip `p == s`, assert no other `in_base` parent" logic.
    var result = try causal.compactGraph(gpa, &self.history, in_base, s);
    errdefer result.graph.deinit(gpa);
    gpa.free(result.lv_map); // TextDoc has no lv-indexed state to remap

    // Commit.
    const head_id = self.history.idOf(s);
    const head_name = try gpa.dupe(u8, self.history.agentName(head_id.agent));
    defer gpa.free(head_name);
    self.history.deinit(gpa);
    self.history = result.graph;
    gpa.free(self.base_bytes);
    gpa.free(self.base_version);
    self.base_bytes = new_base_bytes;
    self.base_scalars = new_base_scalars;
    self.base_version = new_base_version;
    self.base_head = .{ .agent = self.history.findAgent(head_name).?, .seq = head_id.seq };
    // Every retained event was just renumbered (`lv_map`); the cache's
    // lv-indexed arrays point at the old space. Next merge rebuilds it
    // against the new (smaller) base_scalars.
    self.merge_walk.reset(gpa);
}

// ── Merge ───────────────────────────────────────────────────────────

/// Integrate encoded remote events: updates the graph, applies the
/// transformed effects to the rope, and returns the byte-space edit
/// stream (caller owns; shift your anchors through it in order).
/// Duplicate events are skipped; causally incomplete batches (including
/// references into compacted interior history, and compacted batches
/// whose base we do not share) are rejected whole with
/// `error.MissingDependency`; malformed or malicious batches with
/// `error.Corrupt` — in both cases the document is untouched. An empty
/// document bootstraps from a compacted batch, adopting its base.
pub fn merge(self: *TextDoc, gpa: Allocator, bytes: []const u8) MergeError![]Edit {
    var dec = try Decoder.init(gpa, bytes);
    defer dec.deinit(gpa);

    // Base compatibility (before any mutation). Version-only base
    // sections (bytes == null) can never seed a bootstrap.
    const bootstrap = dec.base != null and dec.base.?.bytes != null and
        self.base_version.len == 0 and
        self.history.eventCount() == 0 and self.rope.isEmpty();
    if (dec.base) |b| {
        if (!bootstrap and !std.mem.eql(u8, self.base_version, b.version)) {
            return error.MissingDependency;
        }
    }

    // Register batch agents; compute effective watermarks (prospective
    // for a bootstrap — applied only after validation).
    const aids = try gpa.alloc(AgentId, dec.names.items.len);
    defer gpa.free(aids);
    const eff_base = try gpa.alloc(u64, dec.names.items.len);
    defer gpa.free(eff_base);
    for (dec.names.items, aids, eff_base, dec.seq_bases.items) |name, *aid, *eff, batch_base| {
        aid.* = try self.history.registerAgent(gpa, name);
        const have = self.history.agents.items[@intFromEnum(aid.*)].seq_base;
        if (bootstrap) {
            eff.* = batch_base;
        } else if (batch_base != 0 and batch_base != have) {
            // Same base implies identical watermarks; v1 batches carry 0.
            return error.Corrupt;
        } else {
            eff.* = have;
        }
    }

    // The base head an incoming compacted-parent reference may name.
    var batch_head: ?EventId = self.base_head;
    if (bootstrap) {
        const entry = try versionSingleEntry(dec.base.?.version);
        const agent = self.history.findAgent(entry.name) orelse return error.Corrupt;
        batch_head = .{ .agent = agent, .seq = entry.seq };
    }

    try dec.validate(gpa, self, aids, eff_base, batch_head);

    // Commit the bootstrap: adopt base fields, watermarks, and the rope.
    if (bootstrap) {
        const b = dec.base.?;
        const btext = b.bytes.?;
        if (std.unicode.utf8CountCodepoints(btext) catch null) |c| {
            if (c != b.scalars) return error.Corrupt;
        } else return error.Corrupt;
        self.base_bytes = try gpa.dupe(u8, btext);
        self.base_version = try gpa.dupe(u8, b.version);
        self.base_scalars = b.scalars;
        self.base_head = batch_head;
        for (aids, eff_base) |aid, eff| {
            self.history.agents.items[@intFromEnum(aid)].seq_base = eff;
        }
        self.rope = try Rope.fromSlice(gpa, btext);
    }

    // Graph phase: add events + replay, atomically — on any failure the
    // graph rolls back wholesale and the rope was never touched (a
    // failed bootstrap leaves a coherent base-only document).
    var scalar_edits: std.ArrayList(ScalarEdit) = .empty;
    defer scalar_edits.deinit(gpa);
    const any_new = try self.historyPhase(gpa, &dec, aids, &scalar_edits);
    if (!any_new) return try gpa.alloc(Edit, 0);

    // Apply to the rope, converting scalar → byte space, coalescing runs.
    var edits: std.ArrayList(Edit) = .empty;
    errdefer edits.deinit(gpa);
    var buf: [4]u8 = undefined;
    const holey = self.holes.items.len != 0;
    for (scalar_edits.items) |se| {
        switch (se) {
            .ins => |ins| {
                const off = if (holey) self.insByteOffset(ins.pos) else self.rope.scalarToOffset(ins.pos);
                const len = std.unicode.utf8Encode(ins.ch, &buf) catch unreachable;
                _ = try self.rope.insert(gpa, off, buf[0..len]);
                if (holey) self.shiftHoles(off, len, 0);
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
                const range = if (holey) self.delByteRange(pos) else Range{
                    .start = self.rope.scalarToOffset(pos),
                    .end = self.rope.scalarToOffset(pos + 1),
                };
                const start = range.start;
                const end = range.end;
                _ = try self.rope.delete(gpa, range);
                if (holey) self.shiftHoles(start, 0, end - start);
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

// ── Hole-aware scalar ⇄ byte mapping (merge application) ────────────
// Global scalar positions count unrealized base scalars; the rope's
// scalar metric counts realized content only. Positions interior to a
// hole are excluded beforehand (`checkHoleConflicts`), so mapping only
// has to subtract hole scalars and pin boundary cases to the hole edge.

/// Byte offset for inserting at global scalar position `pos`.
fn insByteOffset(self: *const TextDoc, pos: u64) usize {
    var acc: u64 = 0; // hole scalars strictly before pos
    for (self.holes.items) |h| {
        const start = @as(u64, self.rope.offsetToScalar(h.cur_offset)) + acc;
        if (pos <= start) {
            if (pos == start) return h.cur_offset; // insert before the hole
            break;
        }
        acc += h.scalars; // non-interior ⇒ pos ≥ start + scalars
    }
    return self.rope.scalarToOffset(@intCast(pos - acc));
}

/// Byte range of the (realized) scalar at global position `pos`,
/// clipped so it never swallows a neighboring hole's bytes.
fn delByteRange(self: *const TextDoc, pos: u64) Range {
    var acc: u64 = 0;
    var clamp: ?usize = null; // first hole after the target
    for (self.holes.items) |h| {
        const start = @as(u64, self.rope.offsetToScalar(h.cur_offset)) + acc;
        if (pos < start) {
            clamp = h.cur_offset;
            break;
        }
        acc += h.scalars;
    }
    const rs: usize = @intCast(pos - acc);
    const start = self.rope.scalarToOffset(rs);
    var end = self.rope.scalarToOffset(rs + 1);
    if (clamp) |c| end = @min(end, c);
    return .{ .start = start, .end = end };
}

/// Reject (whole-batch) any transformed edit that lands interior to an
/// unrealized base span: the caller realizes the span and merges again.
fn checkHoleConflicts(self: *const TextDoc, gpa: Allocator, scalar_edits: []const ScalarEdit) MergeError!void {
    if (self.holes.items.len == 0) return;
    // Global scalar start of each hole, tracked through the edit stream.
    const starts = try gpa.alloc(u64, self.holes.items.len);
    defer gpa.free(starts);
    var acc: u64 = 0;
    for (self.holes.items, starts) |h, *s| {
        s.* = @as(u64, self.rope.offsetToScalar(h.cur_offset)) + acc;
        acc += h.scalars;
    }
    for (scalar_edits) |se| switch (se) {
        .ins => |ins| for (self.holes.items, starts) |h, *s| {
            if (ins.pos > s.* and ins.pos < s.* + h.scalars) return error.Unrealized;
            if (ins.pos <= s.*) s.* += 1;
        },
        .del => |pos| for (self.holes.items, starts) |h, *s| {
            if (pos >= s.* and pos < s.* + h.scalars) return error.Unrealized;
            if (pos < s.*) s.* -= 1;
        },
    };
}

/// Add the batch to the graph and replay. Atomic: on any error the graph
/// reverts to its pre-batch state. Returns whether any new events were
/// integrated.
fn historyPhase(
    self: *TextDoc,
    gpa: Allocator,
    dec: *const Decoder,
    aids: []const AgentId,
    scalar_edits: *std.ArrayList(ScalarEdit),
) MergeError!bool {
    const pre_events = self.history.events.items.len;
    const pre_pool = self.history.parents_pool.items.len;
    const pre_frontier = try gpa.dupe(Lv, self.history.frontier.items);
    defer gpa.free(pre_frontier);
    const pre_seq_lens = try gpa.alloc(usize, self.history.agents.items.len);
    defer gpa.free(pre_seq_lens);
    for (self.history.agents.items, pre_seq_lens) |a, *len| len.* = a.lv_by_seq.items.len;
    errdefer self.rollbackHistory(pre_events, pre_pool, pre_frontier, pre_seq_lens);

    const first_new: Lv = @intCast(self.history.eventCount());
    var any_new = false;
    for (dec.events.items) |ev| {
        const id: EventId = .{ .agent = aids[ev.agent_idx], .seq = ev.seq };
        if (self.history.isKnown(id)) continue; // duplicate or compacted
        var parent_lvs: std.ArrayList(Lv) = .empty;
        defer parent_lvs.deinit(gpa);
        for (dec.parentsOf(ev)) |pref| {
            const pid: EventId = .{ .agent = aids[pref.agent_idx], .seq = pref.seq };
            if (self.history.lvOf(pid)) |plv| {
                try parent_lvs.append(gpa, plv);
            }
            // else: validated to be the base head — implicit.
        }
        _ = try self.history.add(gpa, id, parent_lvs.items, ev.op);
        any_new = true;
    }
    if (!any_new) return false;

    // The walker cache resumes from wherever the last successful merge
    // (or, before that, initBase) left it — any local edits recorded
    // since then replay silently alongside this batch's new events, in
    // the same pass. A failure below (bad batch, OOM, a hole conflict)
    // discards the cache rather than trying to unwind it: `s`'s FugueMax
    // arena mutates existing entries' `pos_of` in place on insert, so a
    // length-only rollback would leave it inconsistent. Rare path;
    // the next successful merge just rebuilds from genesis once.
    errdefer self.merge_walk.reset(gpa);
    if (self.merge_walk.walked == 0 and self.base_scalars > 0) {
        try self.merge_walk.initBase(gpa, self.base_scalars);
    }
    try self.merge_walk.replayAll(gpa, &self.history, @intCast(self.history.eventCount()), first_new, null, scalar_edits);
    // Edits landing inside unrealized base spans reject the whole batch
    // (graph rolls back): realize, then merge again.
    try self.checkHoleConflicts(gpa, scalar_edits.items);
    return true;
}

fn rollbackHistory(
    self: *TextDoc,
    pre_events: usize,
    pre_pool: usize,
    pre_frontier: []const Lv,
    pre_seq_lens: []const usize,
) void {
    self.history.events.items.len = pre_events;
    self.history.parents_pool.items.len = pre_pool;
    // Agents registered during decode may outnumber pre_seq_lens; their
    // seq lists were empty before the batch.
    for (self.history.agents.items, 0..) |*a, i| {
        a.lv_by_seq.items.len = if (i < pre_seq_lens.len) pre_seq_lens[i] else 0;
    }
    self.history.frontier.clearRetainingCapacity();
    // Capacity never shrinks, so this cannot fail.
    self.history.frontier.appendSliceAssumeCapacity(pre_frontier);
}

// ── Wire format ─────────────────────────────────────────────────────
// v1: "stg" 0x01
//   uv agent_count, per agent: uv name_len, name
//   uv event_count, per event:
//     uv agent_idx (batch table), uv seq
//     uv parent_count, per parent: uv agent_idx, uv seq
//     u8 tag (0 = ins, 1 = del), uv pos, [uv ch if ins]
// v2 (emitted when compacted): "stg" 0x02, then
//   uv base_version_len, bytes (a version token)
//   uv base_scalars, uv base_bytes_len, bytes (UTF-8)
//   agents as v1 but each entry appends: uv seq_base
//   events as v1 (parent lists may be empty = based on the base)
// v3 (run-RLE, `WireFormat.rle`): "stg" 0x03, then
//   u8 flags (0: no base, 1: full base section, 2: version-only base —
//     partial peers sync same-base but cannot seed bootstraps)
//   [base section: uv version_len, token; full adds uv scalars,
//     uv bytes_len, bytes]
//   agents as v2 (seq_base always present; 0 when uncompacted)
//   uv frame_count, per frame: u8 tag, then
//     uv agent_idx, uv seq (of the frame's FIRST unit),
//     uv parent_count, per parent: uv agent_idx, uv seq
//       (parents of the first unit; run interiors chain implicitly)
//     tag 0 unit ins: uv pos, uv ch
//     tag 1 unit del: uv pos
//     tag 2 ins run:  uv count (≥2), uv base_pos,
//                     u8 dir (0: pos ascends +1/unit, 1: pos constant),
//                     uv ch × count
//     tag 3 del run:  uv count (≥2), uv pos (constant — each unit
//                     deletes at the same position, the next scalar
//                     sliding into place)
//   Runs decode back to unit events: unit i has seq+i, single parent
//   (agent, seq+i-1). The graph model never sees a run.
// All integers unsigned LEB128; versioned by the magic's trailing byte.

/// An encodable insert run: `len` events starting at `lvs[i]`, positions
/// following `dir`.
const InsRun = struct { len: usize, dir: u8 };

fn encodeEvents(
    self: *const TextDoc,
    gpa: Allocator,
    lvs: []const Lv,
    format: WireFormat,
) Allocator.Error![]u8 {
    const compacted = self.base_version.len > 0;
    const partial = self.holes.items.len != 0;
    const rle = format == .rle;
    assert(rle or !partial); // .unit + partial guarded by callers
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    if (rle) {
        try out.appendSlice(gpa, wire_magic_v3);
        const flags: u8 = if (!compacted) 0 else if (partial) 2 else 1;
        try out.append(gpa, flags);
    } else {
        try out.appendSlice(gpa, if (compacted) wire_magic_v2 else wire_magic_v1);
    }
    if (compacted) {
        try putUv(gpa, &out, self.base_version.len);
        try out.appendSlice(gpa, self.base_version);
        if (!partial) {
            // A partial base ships its version only: enough to sync with
            // same-base peers, never enough to bootstrap from.
            try putUv(gpa, &out, self.base_scalars);
            try putUv(gpa, &out, self.base_bytes.len);
            try out.appendSlice(gpa, self.base_bytes);
        }
    }

    // Batch agent table: every agent appearing as author or parent.
    var table: std.ArrayList(AgentId) = .empty;
    defer table.deinit(gpa);
    for (lvs) |lv| {
        try tableAdd(gpa, &table, self.history.idOf(lv).agent);
        for (self.history.parentsOf(lv)) |p| {
            try tableAdd(gpa, &table, self.history.idOf(p).agent);
        }
    }
    if (compacted) {
        if (self.base_head) |h| try tableAdd(gpa, &table, h.agent);
    }
    try putUv(gpa, &out, table.items.len);
    for (table.items) |aid| {
        const name = self.history.agentName(aid);
        try putUv(gpa, &out, name.len);
        try out.appendSlice(gpa, name);
        if (compacted or rle) {
            try putUv(gpa, &out, self.history.agents.items[@intFromEnum(aid)].seq_base);
        }
    }

    if (!rle) {
        try putUv(gpa, &out, lvs.len);
        for (lvs) |lv| {
            const id = self.history.idOf(lv);
            try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
            try putUv(gpa, &out, id.seq);
            try self.writeParents(gpa, &out, table.items, lv);
            switch (self.history.opOf(lv)) {
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

    // RLE: count frames first (greedy run detection is deterministic),
    // then emit.
    var frame_count: usize = 0;
    {
        var i: usize = 0;
        while (i < lvs.len) : (frame_count += 1) {
            i += switch (self.history.opOf(lvs[i])) {
                .ins => self.insRunAt(lvs, i).len,
                .del => self.delRunAt(lvs, i),
            };
        }
    }
    try putUv(gpa, &out, frame_count);
    var i: usize = 0;
    while (i < lvs.len) {
        const first = lvs[i];
        const id = self.history.idOf(first);
        switch (self.history.opOf(first)) {
            .ins => |ins| {
                const run = self.insRunAt(lvs, i);
                try out.append(gpa, if (run.len >= 2) 2 else 0);
                try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
                try putUv(gpa, &out, id.seq);
                try self.writeParents(gpa, &out, table.items, first);
                if (run.len >= 2) {
                    try putUv(gpa, &out, run.len);
                    try putUv(gpa, &out, ins.pos);
                    try out.append(gpa, run.dir);
                    for (lvs[i..][0..run.len]) |lv| {
                        try putUv(gpa, &out, self.history.opOf(lv).ins.ch);
                    }
                } else {
                    try putUv(gpa, &out, ins.pos);
                    try putUv(gpa, &out, ins.ch);
                }
                i += run.len;
            },
            .del => |pos| {
                const len = self.delRunAt(lvs, i);
                try out.append(gpa, if (len >= 2) 3 else 1);
                try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
                try putUv(gpa, &out, id.seq);
                try self.writeParents(gpa, &out, table.items, first);
                if (len >= 2) try putUv(gpa, &out, len);
                try putUv(gpa, &out, pos);
                i += len;
            },
        }
    }
    return out.toOwnedSlice(gpa);
}

/// Parent list of the first unit of a frame. Events based directly on
/// the base re-encode their implicit parent as the base head, so
/// uncompacted-era decoders (and validation) see an explicit dependency.
fn writeParents(
    self: *const TextDoc,
    gpa: Allocator,
    out: *std.ArrayList(u8),
    table: []const AgentId,
    lv: Lv,
) Allocator.Error!void {
    const parents = self.history.parentsOf(lv);
    if (parents.len == 0 and self.base_head != null) {
        const h = self.base_head.?;
        try putUv(gpa, out, 1);
        try putUv(gpa, out, tableIndexOf(table, h.agent));
        try putUv(gpa, out, h.seq);
        return;
    }
    try putUv(gpa, out, parents.len);
    for (parents) |p| {
        const pid = self.history.idOf(p);
        try putUv(gpa, out, tableIndexOf(table, pid.agent));
        try putUv(gpa, out, pid.seq);
    }
}

/// Whether `lvs[j]` extends a run whose previous member is `lvs[j-1]`:
/// same agent, next seq, and its sole parent is that previous event.
fn chains(self: *const TextDoc, lvs: []const Lv, j: usize) bool {
    const prev = lvs[j - 1];
    const cur = lvs[j];
    const prev_id = self.history.idOf(prev);
    const cur_id = self.history.idOf(cur);
    if (cur_id.agent != prev_id.agent or cur_id.seq != prev_id.seq + 1) return false;
    const parents = self.history.parentsOf(cur);
    return parents.len == 1 and parents[0] == prev;
}

/// Maximal insert run starting at `lvs[i]`. Direction is fixed by the
/// second member: ascending (typing forward) or constant position.
fn insRunAt(self: *const TextDoc, lvs: []const Lv, i: usize) InsRun {
    const first_pos = self.history.opOf(lvs[i]).ins.pos;
    var dir: u8 = 0;
    var j = i + 1;
    while (j < lvs.len) : (j += 1) {
        if (!self.chains(lvs, j)) break;
        const op = self.history.opOf(lvs[j]);
        if (op != .ins) break;
        const expect_asc = first_pos + (j - i);
        if (j == i + 1) {
            if (op.ins.pos == expect_asc) {
                dir = 0;
            } else if (op.ins.pos == first_pos) {
                dir = 1;
            } else break;
        } else {
            const expect = if (dir == 0) expect_asc else first_pos;
            if (op.ins.pos != expect) break;
        }
    }
    return .{ .len = j - i, .dir = dir };
}

/// Maximal delete run starting at `lvs[i]`: constant position (each
/// unit deletes at the same offset — the local `delete` pattern).
fn delRunAt(self: *const TextDoc, lvs: []const Lv, i: usize) usize {
    const pos = self.history.opOf(lvs[i]).del;
    var j = i + 1;
    while (j < lvs.len) : (j += 1) {
        if (!self.chains(lvs, j)) break;
        const op = self.history.opOf(lvs[j]);
        if (op != .del or op.del != pos) break;
    }
    return j - i;
}

const tableAdd = core.tableAdd;
const tableIndexOf = core.tableIndexOf;

const ParentRef = struct { agent_idx: u32, seq: u64 };

const DecodedEvent = struct {
    agent_idx: u32,
    seq: u64,
    parents_start: u32,
    parents_len: u32,
    op: TextOp,
};

const BaseSection = struct {
    version: []const u8, // borrowed from the input batch
    scalars: usize,
    /// Null for version-only sections (from partially realized peers) —
    /// enough to sync same-base, never enough to bootstrap from.
    bytes: ?[]const u8, // borrowed from the input batch
};

/// Pure parser: no document mutation. Names and base slices borrow from
/// the input bytes.
const Decoder = struct {
    base: ?BaseSection = null,
    names: std.ArrayList([]const u8) = .empty,
    seq_bases: std.ArrayList(u64) = .empty,
    events: std.ArrayList(DecodedEvent) = .empty,
    parents_pool: std.ArrayList(ParentRef) = .empty,

    fn parentsOf(self: *const Decoder, ev: DecodedEvent) []const ParentRef {
        return self.parents_pool.items[ev.parents_start..][0..ev.parents_len];
    }

    fn deinit(self: *Decoder, gpa: Allocator) void {
        self.names.deinit(gpa);
        self.seq_bases.deinit(gpa);
        self.events.deinit(gpa);
        self.parents_pool.deinit(gpa);
    }

    /// Decoded-unit ceiling per batch. Run frames expand without
    /// consuming input bytes, so a hostile batch could otherwise claim
    /// unbounded counts from a handful of bytes.
    const max_batch_units = 1 << 26;

    fn init(gpa: Allocator, bytes: []const u8) MergeError!Decoder {
        var self: Decoder = .{};
        errdefer self.deinit(gpa);
        var cur: []const u8 = bytes;

        if (cur.len < wire_magic_v1.len or !std.mem.startsWith(u8, cur, "stg")) return error.Corrupt;
        const wire_version = cur[3];
        if (wire_version < 1 or wire_version > 3) return error.Corrupt;
        cur = cur[wire_magic_v1.len..];

        var base_kind: enum { none, full, version_only } = if (wire_version == 2) .full else .none;
        if (wire_version == 3) {
            if (cur.len == 0) return error.Corrupt;
            base_kind = switch (cur[0]) {
                0 => .none,
                1 => .full,
                2 => .version_only,
                else => return error.Corrupt,
            };
            cur = cur[1..];
        }
        const has_seq_base = wire_version >= 2;

        if (base_kind != .none) {
            const vlen = try getUv(&cur);
            if (vlen == 0 or vlen > cur.len) return error.Corrupt;
            const vtoken = cur[0..vlen];
            cur = cur[vlen..];
            _ = try versionSingleEntry(vtoken); // must be a single head
            if (base_kind == .full) {
                const scalars = try getUv(&cur);
                const blen = try getUv(&cur);
                if (blen > cur.len) return error.Corrupt;
                if (scalars > blen) return error.Corrupt;
                const btext = cur[0..blen];
                cur = cur[blen..];
                if (!std.unicode.utf8ValidateSlice(btext)) return error.Corrupt;
                self.base = .{ .version = vtoken, .scalars = @intCast(scalars), .bytes = btext };
            } else {
                self.base = .{ .version = vtoken, .scalars = 0, .bytes = null };
            }
        }

        const agent_count = try getUv(&cur);
        if (agent_count > 1 << 20) return error.Corrupt;
        for (0..agent_count) |_| {
            const nlen = try getUv(&cur);
            if (nlen == 0 or nlen > 4096 or nlen > cur.len) return error.Corrupt;
            try self.names.append(gpa, cur[0..nlen]);
            cur = cur[nlen..];
            try self.seq_bases.append(gpa, if (has_seq_base) try getUv(&cur) else 0);
        }

        if (wire_version < 3) {
            const event_count = try getUv(&cur);
            for (0..event_count) |_| {
                const aidx = try getUv(&cur);
                if (aidx >= self.names.items.len) return error.Corrupt;
                const seq = try getUv(&cur);
                const pcount = try getUv(&cur);
                if (pcount > 1 << 16) return error.Corrupt;
                const pstart: u32 = @intCast(self.parents_pool.items.len);
                for (0..pcount) |_| {
                    const paidx = try getUv(&cur);
                    if (paidx >= self.names.items.len) return error.Corrupt;
                    const pseq = try getUv(&cur);
                    try self.parents_pool.append(gpa, .{ .agent_idx = @intCast(paidx), .seq = pseq });
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
                    .agent_idx = @intCast(aidx),
                    .seq = seq,
                    .parents_start = pstart,
                    .parents_len = @intCast(pcount),
                    .op = op,
                });
            }
            return self;
        }

        try self.initV3(gpa, &cur);
        return self;
    }

    /// v3 body: frames, runs expanded back to unit events. Split from
    /// `init` to keep the hot v1/v2 loop compact.
    fn initV3(self: *Decoder, gpa: Allocator, cur_: *[]const u8) MergeError!void {
        var cur = cur_.*;
        defer cur_.* = cur;
        const frame_count = try getUv(&cur);
        for (0..frame_count) |_| {
            if (cur.len == 0) return error.Corrupt;
            const tag = cur[0];
            cur = cur[1..];
            const aidx_u = try getUv(&cur);
            if (aidx_u >= self.names.items.len) return error.Corrupt;
            const aidx: u32 = @intCast(aidx_u);
            const seq = try getUv(&cur);
            const pstart = try self.readParents(gpa, &cur);
            const pcount: u32 = @intCast(self.parents_pool.items.len - pstart);
            switch (tag) {
                0, 1 => {
                    const op: TextOp = if (tag == 0)
                        .{ .ins = .{ .pos = try getUv(&cur), .ch = try getCodepoint(&cur) } }
                    else
                        .{ .del = try getUv(&cur) };
                    try self.appendUnit(gpa, aidx, seq, pstart, pcount, op);
                },
                2 => {
                    const count = try getUv(&cur);
                    if (count < 2 or count > max_batch_units) return error.Corrupt;
                    const base_pos = try getUv(&cur);
                    _ = std.math.add(u64, seq, count) catch return error.Corrupt;
                    _ = std.math.add(u64, base_pos, count) catch return error.Corrupt;
                    if (cur.len == 0) return error.Corrupt;
                    const dir = cur[0];
                    cur = cur[1..];
                    if (dir > 1) return error.Corrupt;
                    for (0..@intCast(count)) |k| {
                        const ch = try getCodepoint(&cur);
                        const pos = if (dir == 0) base_pos + k else base_pos;
                        if (k == 0) {
                            try self.appendUnit(gpa, aidx, seq, pstart, pcount, .{ .ins = .{ .pos = pos, .ch = ch } });
                        } else {
                            try self.appendChained(gpa, aidx, seq + k, .{ .ins = .{ .pos = pos, .ch = ch } });
                        }
                    }
                },
                3 => {
                    const count = try getUv(&cur);
                    if (count < 2 or count > max_batch_units) return error.Corrupt;
                    _ = std.math.add(u64, seq, count) catch return error.Corrupt;
                    const pos = try getUv(&cur);
                    try self.appendUnit(gpa, aidx, seq, pstart, pcount, .{ .del = pos });
                    for (1..@intCast(count)) |k| {
                        try self.appendChained(gpa, aidx, seq + k, .{ .del = pos });
                    }
                },
                else => return error.Corrupt,
            }
            if (self.events.items.len > max_batch_units) return error.Corrupt;
        }
    }

    /// Parse a parent list into the pool; returns its start index.
    fn readParents(self: *Decoder, gpa: Allocator, cur: *[]const u8) MergeError!u32 {
        const pcount = try getUv(cur);
        if (pcount > 1 << 16) return error.Corrupt;
        const pstart: u32 = @intCast(self.parents_pool.items.len);
        for (0..pcount) |_| {
            const paidx = try getUv(cur);
            if (paidx >= self.names.items.len) return error.Corrupt;
            const pseq = try getUv(cur);
            try self.parents_pool.append(gpa, .{ .agent_idx = @intCast(paidx), .seq = pseq });
        }
        return pstart;
    }

    fn getCodepoint(cur: *[]const u8) error{Corrupt}!u21 {
        const ch = try getUv(cur);
        if (ch > std.math.maxInt(u21) or !std.unicode.utf8ValidCodepoint(@intCast(ch)))
            return error.Corrupt;
        return @intCast(ch);
    }

    fn appendUnit(
        self: *Decoder,
        gpa: Allocator,
        aidx: u32,
        seq: u64,
        pstart: u32,
        pcount: u32,
        op: TextOp,
    ) Allocator.Error!void {
        try self.events.append(gpa, .{
            .agent_idx = aidx,
            .seq = seq,
            .parents_start = pstart,
            .parents_len = pcount,
            .op = op,
        });
    }

    /// A run-interior unit: sole parent is its predecessor in the run.
    fn appendChained(self: *Decoder, gpa: Allocator, aidx: u32, seq: u64, op: TextOp) Allocator.Error!void {
        const pstart: u32 = @intCast(self.parents_pool.items.len);
        try self.parents_pool.append(gpa, .{ .agent_idx = aidx, .seq = seq - 1 });
        try self.appendUnit(gpa, aidx, seq, pstart, 1, op);
    }

    /// Whole-batch causal validation before any graph mutation:
    /// per-agent contiguity against effective watermarks, and parent
    /// resolvability (stored, earlier-in-batch, or the base head —
    /// references into compacted interior are missing dependencies).
    ///
    /// "Earlier in batch" is one contiguous seen-range per agent: a
    /// causally closed batch has per-agent ascending contiguous seqs, so
    /// the range is exact for honest encoders; adversarial orderings
    /// merely fail to extend it and get rejected. O(events + parents).
    fn validate(
        self: *const Decoder,
        gpa: Allocator,
        doc: *const TextDoc,
        aids: []const AgentId,
        eff_base: []const u64,
        batch_head: ?EventId,
    ) (Allocator.Error || error{MissingDependency})!void {
        // Per agent, the batch-seen range [first, next); empty when equal.
        const Seen = struct { first: u64 = 0, next: u64 = 0 };
        const seen = try gpa.alloc(Seen, self.names.items.len);
        defer gpa.free(seen);
        @memset(seen, .{});
        const inRange = struct {
            fn inRange(s: Seen, seq: u64) bool {
                return seq >= s.first and seq < s.next;
            }
        }.inRange;

        for (self.events.items) |ev| {
            defer {
                const s = &seen[ev.agent_idx];
                if (s.first == s.next) {
                    s.* = .{ .first = ev.seq, .next = ev.seq + 1 };
                } else if (ev.seq == s.next) s.next += 1;
            }
            const agent = aids[ev.agent_idx];
            const stored = doc.history.agents.items[@intFromEnum(agent)].lv_by_seq.items.len;
            const next = eff_base[ev.agent_idx] + stored;
            if (ev.seq < next) continue; // duplicate or compacted
            const contiguous = ev.seq == next or
                (ev.seq > 0 and inRange(seen[ev.agent_idx], ev.seq - 1));
            if (!contiguous) return error.MissingDependency;
            for (self.parentsOf(ev)) |pref| {
                const pagent = aids[pref.agent_idx];
                const pid: EventId = .{ .agent = pagent, .seq = pref.seq };
                if (doc.history.lvOf(pid) != null) continue;
                if (inRange(seen[pref.agent_idx], pref.seq)) continue;
                if (batch_head) |h| {
                    if (h.agent == pagent and h.seq == pref.seq) continue;
                }
                return error.MissingDependency;
            }
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
