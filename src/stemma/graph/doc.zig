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
const base_lv = walker_mod.base_lv;
const Rope = rope_mod.Rope;
const Range = geometry.Range;
const Edit = geometry.Edit;

const wire_magic_v1 = "stg\x01";
const wire_magic_v2 = "stg\x02";
const version_magic = "stv\x01";

pub const TextDoc = struct {
    rope: Rope = .empty,
    graph: Graph = .empty,
    agent: ?AgentId = null,

    /// Frozen compacted pre-history (UTF-8), or empty.
    base_bytes: []u8 = &.{},
    base_scalars: usize = 0,
    /// The opaque version token identifying the base; docs can only sync
    /// when their bases are identical (or one side bootstraps).
    base_version: []u8 = &.{},
    /// The single stable head the base was compacted at.
    base_head: ?EventId = null,

    pub const empty: TextDoc = .{};

    pub const MergeError = Allocator.Error || error{ Corrupt, MissingDependency };
    pub const CompactError = MergeError || error{NotCompactable};
    pub const AnchorError = Allocator.Error || error{ Corrupt, MissingDependency, Compacted };

    pub fn deinit(self: *TextDoc, gpa: Allocator) void {
        self.rope.deinit(gpa);
        self.graph.deinit(gpa);
        gpa.free(self.base_bytes);
        gpa.free(self.base_version);
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

    /// Decode a version token to known Lvs. `strict` errors on entries we
    /// have not stored (`MissingDependency`); lenient mode skips them.
    fn decodeVersion(
        self: *const TextDoc,
        gpa: Allocator,
        token: []const u8,
        strict: bool,
        out: *std.ArrayList(Lv),
    ) MergeError!void {
        var cur = token;
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
            const lv: ?Lv = if (self.graph.findAgent(name)) |agent|
                self.graph.lvOf(.{ .agent = agent, .seq = seq })
            else
                null;
            if (lv) |v| {
                try out.append(gpa, v);
            } else if (strict) {
                return error.MissingDependency;
            }
        }
    }

    /// Parse the single (name, seq) entry of a version token.
    fn versionSingleEntry(token: []const u8) error{Corrupt}!struct { name: []const u8, seq: u64 } {
        var cur = token;
        if (!std.mem.startsWith(u8, cur, version_magic)) return error.Corrupt;
        cur = cur[version_magic.len..];
        if (try getUv(&cur) != 1) return error.Corrupt;
        const nlen = try getUv(&cur);
        if (nlen == 0 or nlen > cur.len) return error.Corrupt;
        const name = cur[0..nlen];
        cur = cur[nlen..];
        return .{ .name = name, .seq = try getUv(&cur) };
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
        self.decodeVersion(gpa, remote_version, false, &known) catch |e| switch (e) {
            error.MissingDependency => unreachable, // lenient mode
            else => |err| return err,
        };
        var missing = try self.graph.missingFrom(gpa, known.items);
        defer missing.deinit(gpa);
        return self.encodeEvents(gpa, missing.items);
    }

    /// The whole document as its event graph (plus the base snapshot when
    /// compacted) — the durable form.
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

    // ── Time travel ─────────────────────────────────────────────────────

    /// Materialize the document as it was at `version` (a token from
    /// `version()`, ours or a peer's — every entry must be stored by us).
    /// Returns a fresh Rope the caller owns. O(history) replay.
    pub fn materializeAt(self: *const TextDoc, gpa: Allocator, version_token: []const u8) MergeError!Rope {
        var heads: std.ArrayList(Lv) = .empty;
        defer heads.deinit(gpa);
        try self.decodeVersion(gpa, version_token, true, &heads);

        const n = self.graph.eventCount();
        const include = try gpa.alloc(bool, n);
        defer gpa.free(include);
        @memset(include, false);
        var d = try self.graph.diff(gpa, heads.items, &.{});
        defer d.deinit(gpa);
        for (d.a_only.items) |lv| include[lv] = true;

        var w = Walker.init(&self.graph);
        defer w.deinit(gpa);
        if (self.base_scalars > 0) try w.initBase(gpa, self.base_scalars);
        var sink: std.ArrayList(ScalarEdit) = .empty;
        defer sink.deinit(gpa);
        w.replayAll(gpa, @intCast(n), include, &sink) catch |e| switch (e) {
            error.Corrupt => unreachable, // trusted local history
            else => |err| return err,
        };
        assert(sink.items.len == 0); // fully silent

        const base_chars = try self.decodeBaseScalars(gpa);
        defer gpa.free(base_chars);
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        var it = w.aliveIterator();
        var buf: [4]u8 = undefined;
        while (it.next()) |alive| {
            const ch = if (alive.lv == base_lv)
                base_chars[alive.arena]
            else
                self.graph.opOf(alive.lv).ins.ch;
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

    pub const AnchorSide = enum { before, after };

    pub const EventAnchor = struct {
        /// Inserting agent's name; empty = document boundary. Owned by the
        /// caller (`anchorAt` allocates it; free with `gpa.free`).
        agent: []const u8 = "",
        seq: u64 = 0,
        side: AnchorSide = .before,
    };

    /// An identity anchor for the position `byte_offset`. `stickiness`
    /// chooses the attachment: `.before` attaches to the character at the
    /// offset (anchor rides in front of it), `.after` to the character
    /// preceding it. Compacted characters cannot be anchored
    /// (`error.Compacted`). The returned `agent` slice is gpa-owned.
    pub fn anchorAt(
        self: *const TextDoc,
        gpa: Allocator,
        byte_offset: usize,
        stickiness: AnchorSide,
    ) AnchorError!EventAnchor {
        const scalar = self.rope.offsetToScalar(byte_offset);
        const total = self.rope.scalarLen();
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
        var it = w.aliveIterator();
        var i: usize = 0;
        while (it.next()) |alive| : (i += 1) {
            if (i == target_index) {
                if (alive.lv == base_lv) return error.Compacted;
                const id = self.graph.idOf(alive.lv);
                return .{
                    .agent = try gpa.dupe(u8, self.graph.agentName(id.agent)),
                    .seq = id.seq,
                    .side = stickiness,
                };
            }
        }
        unreachable; // target_index < alive count by construction
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

        // One pass: per-arena count of alive items strictly before it.
        const alive_before = try gpa.alloc(u64, w.items.items.len);
        defer gpa.free(alive_before);
        var count: u64 = 0;
        for (w.seq.items) |arena| {
            alive_before[arena] = count;
            if (w.items.items[arena].effect_visible) count += 1;
        }

        for (anchors, out) |a, *o| {
            if (a.agent.len == 0) {
                o.* = switch (a.side) {
                    .before => 0,
                    .after => self.rope.byteLen(),
                };
                continue;
            }
            const agent = self.graph.findAgent(a.agent) orelse return error.MissingDependency;
            const id: EventId = .{ .agent = agent, .seq = a.seq };
            const lv = self.graph.lvOf(id) orelse {
                return if (self.graph.isKnown(id)) error.Compacted else error.MissingDependency;
            };
            if (self.graph.opOf(lv) != .ins) return error.Corrupt;
            const arena = w.item_of.items[lv];
            assert(arena != -1);
            const item = &w.items.items[@intCast(arena)];
            var scalar = alive_before[@intCast(arena)];
            if (item.effect_visible and a.side == .after) scalar += 1;
            o.* = self.rope.scalarToOffset(scalar);
        }
    }

    /// Full silent replay of the whole graph (current state).
    fn silentReplay(self: *const TextDoc, gpa: Allocator) Allocator.Error!Walker {
        var w = Walker.init(&self.graph);
        errdefer w.deinit(gpa);
        if (self.base_scalars > 0) try w.initBase(gpa, self.base_scalars);
        var sink: std.ArrayList(ScalarEdit) = .empty;
        defer sink.deinit(gpa);
        w.replayAll(gpa, @intCast(self.graph.eventCount()), null, &sink) catch |e| switch (e) {
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

        const n = self.graph.eventCount();
        const in_base = try gpa.alloc(bool, n);
        defer gpa.free(in_base);
        @memset(in_base, false);
        var d = try self.graph.diff(gpa, &.{s}, &.{});
        defer d.deinit(gpa);
        for (d.a_only.items) |lv| in_base[lv] = true;

        // Every retained event must sit causally after `s`, reachable only
        // through retained events (or `s` itself).
        for (0..n) |lv| {
            if (in_base[lv]) continue;
            for (self.graph.parentsOf(@intCast(lv))) |p| {
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
        // the base, retained events re-added in causal order.
        var new_graph: Graph = .empty;
        errdefer new_graph.deinit(gpa);
        for (self.graph.agents.items, 0..) |a, i| {
            const name = self.graph.names.items[a.name_start..][0..a.name_len];
            const aid = try new_graph.registerAgent(gpa, name);
            assert(@intFromEnum(aid) == i);
            var compacted: u64 = 0;
            for (a.lv_by_seq.items) |lv| {
                if (in_base[lv]) compacted += 1;
            }
            new_graph.agents.items[i].seq_base = a.seq_base + compacted;
        }
        const lv_map = try gpa.alloc(Lv, n);
        defer gpa.free(lv_map);
        var parent_buf: std.ArrayList(Lv) = .empty;
        defer parent_buf.deinit(gpa);
        for (0..n) |old_lv| {
            if (in_base[old_lv]) continue;
            parent_buf.clearRetainingCapacity();
            for (self.graph.parentsOf(@intCast(old_lv))) |p| {
                if (p == s) continue; // implicit: based on the base
                assert(!in_base[p]);
                try parent_buf.append(gpa, lv_map[p]);
            }
            const e = self.graph.events.items[old_lv];
            lv_map[old_lv] = try new_graph.add(gpa, e.id, parent_buf.items, e.op);
        }

        // Commit.
        const head_id = self.graph.idOf(s);
        const head_name = try gpa.dupe(u8, self.graph.agentName(head_id.agent));
        defer gpa.free(head_name);
        self.graph.deinit(gpa);
        self.graph = new_graph;
        gpa.free(self.base_bytes);
        gpa.free(self.base_version);
        self.base_bytes = new_base_bytes;
        self.base_scalars = new_base_scalars;
        self.base_version = new_base_version;
        self.base_head = .{ .agent = self.graph.findAgent(head_name).?, .seq = head_id.seq };
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

        // Base compatibility (before any mutation).
        const bootstrap = dec.base != null and self.base_version.len == 0 and
            self.graph.eventCount() == 0 and self.rope.isEmpty();
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
            aid.* = try self.graph.registerAgent(gpa, name);
            const have = self.graph.agents.items[@intFromEnum(aid.*)].seq_base;
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
            const agent = self.graph.findAgent(entry.name) orelse return error.Corrupt;
            batch_head = .{ .agent = agent, .seq = entry.seq };
        }

        try dec.validate(self, aids, eff_base, batch_head);

        // Commit the bootstrap: adopt base fields, watermarks, and the rope.
        if (bootstrap) {
            const b = dec.base.?;
            if (std.unicode.utf8CountCodepoints(b.bytes) catch null) |c| {
                if (c != b.scalars) return error.Corrupt;
            } else return error.Corrupt;
            self.base_bytes = try gpa.dupe(u8, b.bytes);
            self.base_version = try gpa.dupe(u8, b.version);
            self.base_scalars = b.scalars;
            self.base_head = batch_head;
            for (aids, eff_base) |aid, eff| {
                self.graph.agents.items[@intFromEnum(aid)].seq_base = eff;
            }
            self.rope = try Rope.fromSlice(gpa, b.bytes);
        }

        // Graph phase: add events + replay, atomically — on any failure the
        // graph rolls back wholesale and the rope was never touched (a
        // failed bootstrap leaves a coherent base-only document).
        var scalar_edits: std.ArrayList(ScalarEdit) = .empty;
        defer scalar_edits.deinit(gpa);
        const any_new = try self.graphPhase(gpa, &dec, aids, &scalar_edits);
        if (!any_new) return try gpa.alloc(Edit, 0);

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

    /// Add the batch to the graph and replay. Atomic: on any error the graph
    /// reverts to its pre-batch state. Returns whether any new events were
    /// integrated.
    fn graphPhase(
        self: *TextDoc,
        gpa: Allocator,
        dec: *const Decoder,
        aids: []const AgentId,
        scalar_edits: *std.ArrayList(ScalarEdit),
    ) MergeError!bool {
        const pre_events = self.graph.events.items.len;
        const pre_pool = self.graph.parents_pool.items.len;
        const pre_frontier = try gpa.dupe(Lv, self.graph.frontier.items);
        defer gpa.free(pre_frontier);
        const pre_seq_lens = try gpa.alloc(usize, self.graph.agents.items.len);
        defer gpa.free(pre_seq_lens);
        for (self.graph.agents.items, pre_seq_lens) |a, *len| len.* = a.lv_by_seq.items.len;
        errdefer self.rollbackGraph(pre_events, pre_pool, pre_frontier, pre_seq_lens);

        const first_new: Lv = @intCast(self.graph.eventCount());
        var any_new = false;
        for (dec.events.items) |ev| {
            const id: EventId = .{ .agent = aids[ev.agent_idx], .seq = ev.seq };
            if (self.graph.isKnown(id)) continue; // duplicate or compacted
            var parent_lvs: std.ArrayList(Lv) = .empty;
            defer parent_lvs.deinit(gpa);
            for (dec.parentsOf(ev)) |pref| {
                const pid: EventId = .{ .agent = aids[pref.agent_idx], .seq = pref.seq };
                if (self.graph.lvOf(pid)) |plv| {
                    try parent_lvs.append(gpa, plv);
                }
                // else: validated to be the base head — implicit.
            }
            _ = try self.graph.add(gpa, id, parent_lvs.items, ev.op);
            any_new = true;
        }
        if (!any_new) return false;

        var w = Walker.init(&self.graph);
        defer w.deinit(gpa);
        if (self.base_scalars > 0) try w.initBase(gpa, self.base_scalars);
        try w.replayAll(gpa, first_new, null, scalar_edits);
        return true;
    }

    fn rollbackGraph(
        self: *TextDoc,
        pre_events: usize,
        pre_pool: usize,
        pre_frontier: []const Lv,
        pre_seq_lens: []const usize,
    ) void {
        self.graph.events.items.len = pre_events;
        self.graph.parents_pool.items.len = pre_pool;
        // Agents registered during decode may outnumber pre_seq_lens; their
        // seq lists were empty before the batch.
        for (self.graph.agents.items, 0..) |*a, i| {
            a.lv_by_seq.items.len = if (i < pre_seq_lens.len) pre_seq_lens[i] else 0;
        }
        self.graph.frontier.clearRetainingCapacity();
        // Capacity never shrinks, so this cannot fail.
        self.graph.frontier.appendSliceAssumeCapacity(pre_frontier);
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
    // All integers unsigned LEB128; versioned by the magic's trailing byte.

    fn encodeEvents(self: *const TextDoc, gpa: Allocator, lvs: []const Lv) Allocator.Error![]u8 {
        const compacted = self.base_version.len > 0;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, if (compacted) wire_magic_v2 else wire_magic_v1);
        if (compacted) {
            try putUv(gpa, &out, self.base_version.len);
            try out.appendSlice(gpa, self.base_version);
            try putUv(gpa, &out, self.base_scalars);
            try putUv(gpa, &out, self.base_bytes.len);
            try out.appendSlice(gpa, self.base_bytes);
        }

        // Batch agent table: every agent appearing as author or parent.
        var table: std.ArrayList(AgentId) = .empty;
        defer table.deinit(gpa);
        for (lvs) |lv| {
            try tableAdd(gpa, &table, self.graph.idOf(lv).agent);
            for (self.graph.parentsOf(lv)) |p| {
                try tableAdd(gpa, &table, self.graph.idOf(p).agent);
            }
        }
        if (compacted) {
            if (self.base_head) |h| try tableAdd(gpa, &table, h.agent);
        }
        try putUv(gpa, &out, table.items.len);
        for (table.items) |aid| {
            const name = self.graph.agentName(aid);
            try putUv(gpa, &out, name.len);
            try out.appendSlice(gpa, name);
            if (compacted) {
                try putUv(gpa, &out, self.graph.agents.items[@intFromEnum(aid)].seq_base);
            }
        }

        try putUv(gpa, &out, lvs.len);
        for (lvs) |lv| {
            const id = self.graph.idOf(lv);
            try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
            try putUv(gpa, &out, id.seq);
            const parents = self.graph.parentsOf(lv);
            const implicit_base = parents.len == 0 and self.base_head != null and lv < self.graph.eventCount();
            if (implicit_base and self.graph.parentsOf(lv).len == 0) {
                // Events based directly on the base re-encode their implicit
                // parent as the base head, so uncompacted-era decoders (and
                // validation) see an explicit dependency.
                const h = self.base_head.?;
                try putUv(gpa, &out, 1);
                try putUv(gpa, &out, tableIndexOf(table.items, h.agent));
                try putUv(gpa, &out, h.seq);
            } else {
                try putUv(gpa, &out, parents.len);
                for (parents) |p| {
                    const pid = self.graph.idOf(p);
                    try putUv(gpa, &out, tableIndexOf(table.items, pid.agent));
                    try putUv(gpa, &out, pid.seq);
                }
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
        bytes: []const u8, // borrowed from the input batch
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

        fn init(gpa: Allocator, bytes: []const u8) MergeError!Decoder {
            var self: Decoder = .{};
            errdefer self.deinit(gpa);
            var cur: []const u8 = bytes;

            const v2 = std.mem.startsWith(u8, cur, wire_magic_v2);
            if (!v2 and !std.mem.startsWith(u8, cur, wire_magic_v1)) return error.Corrupt;
            cur = cur[wire_magic_v1.len..];

            if (v2) {
                const vlen = try getUv(&cur);
                if (vlen == 0 or vlen > cur.len) return error.Corrupt;
                const vtoken = cur[0..vlen];
                cur = cur[vlen..];
                const scalars = try getUv(&cur);
                const blen = try getUv(&cur);
                if (blen > cur.len) return error.Corrupt;
                if (scalars > blen) return error.Corrupt;
                const btext = cur[0..blen];
                cur = cur[blen..];
                if (!std.unicode.utf8ValidateSlice(btext)) return error.Corrupt;
                _ = try versionSingleEntry(vtoken); // must be a single head
                self.base = .{ .version = vtoken, .scalars = @intCast(scalars), .bytes = btext };
            }

            const agent_count = try getUv(&cur);
            if (agent_count > 1 << 20) return error.Corrupt;
            for (0..agent_count) |_| {
                const nlen = try getUv(&cur);
                if (nlen == 0 or nlen > 4096 or nlen > cur.len) return error.Corrupt;
                try self.names.append(gpa, cur[0..nlen]);
                cur = cur[nlen..];
                try self.seq_bases.append(gpa, if (v2) try getUv(&cur) else 0);
            }

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

        /// Whole-batch causal validation before any graph mutation:
        /// per-agent contiguity against effective watermarks, and parent
        /// resolvability (stored, earlier-in-batch, or the base head —
        /// references into compacted interior are missing dependencies).
        fn validate(
            self: *const Decoder,
            doc: *const TextDoc,
            aids: []const AgentId,
            eff_base: []const u64,
            batch_head: ?EventId,
        ) error{MissingDependency}!void {
            for (self.events.items, 0..) |ev, i| {
                const agent = aids[ev.agent_idx];
                const stored = doc.graph.agents.items[@intFromEnum(agent)].lv_by_seq.items.len;
                const next = eff_base[ev.agent_idx] + stored;
                if (ev.seq < next) continue; // duplicate or compacted
                const contiguous = ev.seq == next or
                    (ev.seq > 0 and self.seenEarlier(i, ev.agent_idx, ev.seq - 1));
                if (!contiguous) return error.MissingDependency;
                for (self.parentsOf(ev)) |pref| {
                    const pagent = aids[pref.agent_idx];
                    const pid: EventId = .{ .agent = pagent, .seq = pref.seq };
                    if (doc.graph.lvOf(pid) != null) continue;
                    if (self.seenEarlier(i, pref.agent_idx, pref.seq)) continue;
                    if (batch_head) |h| {
                        if (h.agent == pagent and h.seq == pref.seq) continue;
                    }
                    return error.MissingDependency;
                }
            }
        }

        fn seenEarlier(self: *const Decoder, before: usize, agent_idx: u32, seq: u64) bool {
            for (self.events.items[0..before]) |ev| {
                if (ev.agent_idx == agent_idx and ev.seq == seq) return true;
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
