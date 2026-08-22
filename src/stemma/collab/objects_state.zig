//! Transient replay state for ObjectDoc: the eg-walker prepare/effect
//! discipline generalized to a *tree* of objects — multi-value map
//! registers plus per-object sequences. All sequence ordering comes from
//! the shared FugueMax engine in sequence.zig; this file owns only what is
//! object-specific: per-object dispatch, the MV register semantics, and
//! kill-list bookkeeping for map writes.
//!
//! Map semantics are multi-value ("MV"): a set overwrites exactly the
//! values its author could see (prepare-visible); concurrent sets to the
//! same key all survive as siblings — the honest option in a clockless
//! system. Readers get a deterministic winner plus the full conflict set.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const Sequence = @import("Sequence.zig");
const seq_walker = @import("SeqWalker.zig");
// Each object's SeqWalker sits over a `Lv` space shared with every other
// object and every map register in the tree — a dense array would cost
// memory proportional to the WHOLE document, once per object (see
// `SeqWalker.Storage`). `pub`: `ObjectDoc`'s identity-anchor machinery
// (`objectAnchorAt`/`resolveObjectAnchors`) needs to name this exact type
// to pull one object's already-replayed `SeqWalker` out of `Walker.seqs`.
pub const SeqWalker = seq_walker.SeqWalker(.sparse);
const Lv = causal.Lv;
const EventId = causal.EventId;
const none = Sequence.none;

/// String reference into the owning ObjectDoc's append-only arena.
pub const Str = struct { start: u32, len: u32 };

/// Object identity: the event that created it. `null` obj refs = the root
/// map. DOC-LOCAL (embeds replica-local agent numbering); see
/// ObjectDoc.exportId for portable references.
pub const ObjId = EventId;

pub const ValPayload = union(enum) {
    null_,
    bool_: bool,
    int: i64,
    float: f64,
    str: Str,
    new_map,
    new_list,
    new_text,
};

/// Where a structural node's parent register can point (F3, delta 6 —
/// `stemma-unification.md` §3 step 5, ported from `structure_sketch.zig`'s
/// `NodeRef`): the two permanent roots, or another structural node by
/// portable identity. Wire-portable (`ObjId` = `EventId`).
pub const StructRef = union(enum) {
    root,
    trash,
    node: ObjId,
};

pub const ObjectOp = union(enum) {
    map_set: struct { obj: ?ObjId, key: Str, val: ValPayload },
    map_del: struct { obj: ?ObjId, key: Str },
    list_ins: struct { obj: ObjId, pos: u64, val: ValPayload },
    list_del: struct { obj: ObjId, pos: u64 },
    text_ins: struct { obj: ObjId, pos: u64, ch: u21 },
    text_del: struct { obj: ObjId, pos: u64 },
    /// Allocates a new structural node (identity = this event's own id,
    /// and — see `Walker.resolveObj`'s `.struct_create` case — it also
    /// doubles as an ordinary map object, so `mapSet`/`mapGet` work on it
    /// unmodified) and performs its first parent-register write in the
    /// same event. Mirrors `structure_sketch.zig`'s `StructureOp.create`.
    struct_create: struct { parent: StructRef, order_key: Str },
    /// A subsequent parent-register write against an existing structural
    /// node. Identity-preserving move: exactly one register write, same
    /// shape as `create`'s. Mirrors the sketch's `StructureOp.move`.
    struct_move: struct { node: ObjId, parent: StructRef, order_key: Str },
};

pub const Graph = causal.EventGraph(ObjectOp);

// ── Order keys (F3, delta 6): byte-string fractional midpoints ─────────
// Ported verbatim from `structure_sketch.zig`'s `between` — see that
// file's module doc for the full growth-bound discussion (random
// insertion ~O(log N); adversarial same-gap insertion ~N/8 bytes; a real
// implementation needs periodic sibling-key rebalancing during
// compaction, not attempted here — see `ObjectDoc.compact`'s doc comment
// on why structural ops refuse compaction outright for now).

/// Byte-string fractional midpoint strictly between `a` and `b` (`null` a =
/// -infinity, `null` b = +infinity). Caller owns the returned slice.
pub fn between(gpa: Allocator, a: ?[]const u8, b: ?[]const u8) Allocator.Error![]u8 {
    assert(a == null or b == null or std.mem.order(u8, a.?, b.?) == .lt);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (true) {
        const da: u16 = if (a) |aa| (if (i < aa.len) aa[i] else 0) else 0;
        const db: u16 = if (b) |bb| (if (i < bb.len) bb[i] else 256) else 256;
        if (db - da >= 2) {
            const mid = da + (db - da) / 2;
            try out.append(gpa, @intCast(mid));
            return out.toOwnedSlice(gpa);
        }
        try out.append(gpa, @intCast(da));
        i += 1;
    }
}

/// A text object's compacted pre-history: alive scalar content as of the
/// stable point it was last compacted at (`ObjectDoc.compact`, delta 2,
/// `stemma-unification.md` §3 step 4) — the per-object analog of
/// `TextDoc.base_bytes`/`base_scalars`. Only text objects get one (see
/// `ObjectDoc.zig`'s compaction doc comment for why list content doesn't).
/// Owned by the `ObjectDoc`; `Walker` only borrows it (read-only, to seed
/// a fresh per-object `SeqWalker`'s `initBase`).
pub const TextBase = struct { bytes: []const u8, scalars: usize };
/// Keyed by the text object's PORTABLE creation identity, not its `Lv` —
/// under whole-doc compaction the object's OWN creation event (a
/// `map_set` or `list_ins`) can itself end up compacted away, same as
/// any other event before the stable point (see `ObjectDoc.compact`'s
/// doc comment on why `in_base` can't selectively exempt op kinds). A
/// `Lv`-keyed map would need re-keying on every `compact`; `EventId` never
/// changes, so this one doesn't.
pub const TextBaseMap = std.AutoHashMapUnmanaged(causal.EventId, TextBase);

/// Walker-internal (Lv-space) counterpart to `StructRef` — resolved once
/// per structural event via `Walker.resolveStructRef`.
pub const StructTarget = union(enum) {
    root,
    trash,
    node: Lv,
};

/// Effects emitted for NEW events during replay, in application order.
/// Positions are indices/scalars valid against the state produced by all
/// previously emitted effects.
pub const Effect = union(enum) {
    map_add: struct { obj: ?Lv, key: Str, set_lv: Lv, val: ValPayload },
    map_remove: struct { obj: ?Lv, key: Str, set_lv: Lv },
    list_ins: struct { obj: Lv, index: u64, lv: Lv, val: ValPayload },
    list_del: struct { obj: Lv, index: u64 },
    text_ins: struct { obj: Lv, pos: u64, ch: u21 },
    text_del: struct { obj: Lv, pos: u64 },
    /// A structural node (F3, delta 6) was just created — emitted EAGERLY,
    /// in causal Lv order, from `struct_create`'s own dispatch in `apply`
    /// (unlike `struct_parent` below, which is a deferred post-pass
    /// effect). This has to be its own effect, emitted in-line: a
    /// SAME-BATCH `map_set`/`list_ins`/`text_ins` targeting this node
    /// (always causally AFTER it, hence later in Lv order — see
    /// `Walker.resolveObj`'s `.struct_create` case) needs the node to
    /// already exist by the time IT is processed, and the deferred
    /// `struct_parent` post-pass runs only after the WHOLE main loop
    /// finishes. A `struct_create`'s own placement can never be
    /// cycle-rejected (a brand-new node cannot already be anyone's
    /// ancestor — see `Walker.resolveStructs`'s doc comment), so emitting
    /// its existence eagerly is always sound.
    struct_created: struct { node: Lv },
    /// A structural node's parent-register EFFECTIVE state (winner) has
    /// changed as a result of this replay — see `Walker.emitStructEffects`
    /// for the before/after diff that decides when this fires, and the
    /// module doc / `ObjectDoc.structParent` for the landmine: this is NOT
    /// the same winner rule `map_add`/`map_remove` implement. One entry
    /// per NODE (not per write) — only the FINAL resolved state after this
    /// replay, unlike map's per-event add/remove pairs.
    struct_parent: struct {
        node: Lv,
        parent: StructTarget,
        order_key: Str,
        /// The Lv of the write that IS the effective parent (may be an
        /// earlier, already-superseded write — see `cycle_broken`).
        key_writer: Lv,
        /// Size of the node's causally-maximal register antichain.
        conflict_count: u32,
        /// True iff `key_writer` is NOT a member of that antichain — the
        /// cycle-break survivor case (F3 caveat 1, landmine 1).
        cycle_broken: bool,
    },
};

// ── Per-(object, key) register state ────────────────────────────────────

const RegItem = struct {
    lv: Lv, // the map_set event
    prep_inserted: bool,
    prep_overwritten: u32,
    effect_active: bool,

    fn prepVisible(it: *const RegItem) bool {
        return it.prep_inserted and it.prep_overwritten == 0;
    }
};

const KeySlot = struct {
    key: Str,
    items: std.ArrayList(RegItem) = .empty,
};

const MapReg = struct {
    slots: std.ArrayList(KeySlot) = .empty,

    fn deinit(self: *MapReg, gpa: Allocator) void {
        for (self.slots.items) |*s| s.items.deinit(gpa);
        self.slots.deinit(gpa);
    }
};

// ── The replay state ────────────────────────────────────────────────────

pub const Walker = struct {
    graph: *const Graph,
    strings: []const u8,
    /// This doc's compacted text-object bases, if any (`null` for a doc
    /// that has never compacted — the common case, and every call site
    /// before delta 2). Borrowed; `Walker` never mutates it. Consulted
    /// exactly once per object, the moment its `SeqWalker` is first
    /// created (`getSeqWalker`) — list objects are never present in the
    /// map (see `TextBase`'s doc comment), so this is a no-op lookup for
    /// them, same cost as today.
    text_bases: ?*const TextBaseMap = null,

    /// Object-lv (or `root_key`) → sequence / register state. Each
    /// `SeqWalker` owns its own object's retreat/advance/apply
    /// bookkeeping (see `SeqWalker.zig`) — the shared discipline `Walker`
    /// also drives, per object, alongside the map-register logic below.
    seqs: std.AutoHashMapUnmanaged(Lv, SeqWalker) = .empty,
    maps: std.AutoHashMapUnmanaged(Lv, MapReg) = .empty,
    /// map_set events: index of their own RegItem within the key slot.
    reg_pos_of: std.ArrayList(i32) = .empty,
    /// Kill lists for map_set/map_del: spans into `kills_pool` of RegItem
    /// indices (within the event's key slot) it overwrote.
    kills_start: std.ArrayList(u32) = .empty,
    kills_len: std.ArrayList(u32) = .empty,
    kills_pool: std.ArrayList(u32) = .empty,
    prep_frontier: std.ArrayList(Lv) = .empty,
    /// Per-event Lamport timestamp (F3, delta 6) — filled ONLY when
    /// `emitStructEffects` actually runs (`struct_events.items.len > 0`
    /// AND there's something new to report), by `computeLamport`'s own
    /// whole-graph pass (the SAME shape as `structure_sketch.zig`'s
    /// `computeLamport` — a separate pass, not threaded into the main
    /// per-event loop above). Deliberately NOT computed unconditionally:
    /// the overwhelming majority of replays (every text/map/list-only
    /// doc — W7's degenerate one-node text buffers chief among them) have
    /// zero structural events and must pay zero cost for a mechanism they
    /// never touch. Empty (and never indexed) for such a replay.
    lamport: std.ArrayList(u32) = .empty,
    /// Lv's of every `struct_create`/`struct_move` event visited this
    /// replay, in Lv (causal) order — the candidate pool
    /// `emitStructEffects` sorts into GLOBAL Lamport-canonical order. Kept
    /// separate from `maps`/`seqs` because parent-register resolution is
    /// not a per-key/per-sequence discipline (see landmine 1).
    struct_events: std.ArrayList(Lv) = .empty,

    pub const root_key: Lv = std.math.maxInt(Lv);

    pub const ReplayError = Allocator.Error || error{Corrupt};

    const Names = struct {
        g: *const Graph,
        pub fn agentNameOf(self: @This(), lv: Lv) []const u8 {
            return self.g.agentName(self.g.idOf(lv).agent);
        }
    };

    pub fn init(graph: *const Graph, strings: []const u8) Walker {
        return .{ .graph = graph, .strings = strings };
    }

    /// Same as `init`, plus this doc's compacted text-object bases — every
    /// call site that replays against a doc which may have compacted
    /// (`ObjectDoc.historyPhase`, `ObjectDoc.silentObjectReplay`,
    /// `ObjectDoc.compact`'s own base-materialization pass) must use this
    /// instead of `init`, or a freshly-touched compacted text object's
    /// `SeqWalker` would start with no base at all (empty document,
    /// silently wrong — not an error, so this is worth getting right at
    /// every call site rather than defaulting).
    pub fn initWithBases(graph: *const Graph, strings: []const u8, text_bases: *const TextBaseMap) Walker {
        return .{ .graph = graph, .strings = strings, .text_bases = text_bases };
    }

    pub fn deinit(self: *Walker, gpa: Allocator) void {
        var sit = self.seqs.valueIterator();
        while (sit.next()) |s| s.deinit(gpa);
        self.seqs.deinit(gpa);
        var mit = self.maps.valueIterator();
        while (mit.next()) |m| m.deinit(gpa);
        self.maps.deinit(gpa);
        self.reg_pos_of.deinit(gpa);
        self.kills_start.deinit(gpa);
        self.kills_len.deinit(gpa);
        self.kills_pool.deinit(gpa);
        self.prep_frontier.deinit(gpa);
        self.lamport.deinit(gpa);
        self.struct_events.deinit(gpa);
    }

    fn str(self: *const Walker, s: Str) []const u8 {
        return self.strings[s.start..][0..s.len];
    }

    /// Resolve an op's object reference to its creation Lv (or root_key),
    /// checking existence, causal order, and kind. `expect`: 0 map, 1 list,
    /// 2 text.
    fn resolveObj(self: *const Walker, ref: ?ObjId, at_lv: Lv, comptime expect: u2, untrusted: bool) error{Corrupt}!Lv {
        const id = ref orelse {
            if (expect != 0) {
                if (untrusted) return error.Corrupt;
                unreachable;
            }
            return root_key;
        };
        const creation = self.graph.lvOf(id) orelse {
            if (untrusted) return error.Corrupt;
            unreachable;
        };
        if (creation >= at_lv) {
            if (untrusted) return error.Corrupt;
            unreachable;
        }
        const val: ?ValPayload = switch (self.graph.opOf(creation)) {
            .map_set => |m| m.val,
            .list_ins => |l| l.val,
            // A structural node (F3, delta 6) doubles as an ordinary map
            // object for property storage — its OWN placement in the
            // parent-register tree is a separate concern (see
            // `resolveStructRef`/`resolveStructNode`), but `mapSet`/
            // `mapGet` against it work unmodified once it exists.
            .struct_create => .new_map,
            else => null,
        };
        const ok = if (val) |v| switch (v) {
            .new_map => expect == 0,
            .new_list => expect == 1,
            .new_text => expect == 2,
            else => false,
        } else false;
        if (!ok) {
            if (untrusted) return error.Corrupt;
            unreachable;
        }
        return creation;
    }

    /// Replay the whole graph in Lv order; events at `lv >= first_new`
    /// emit effects. `include` (mirrors `TextDoc.Replay.replayAll`): when
    /// non-null, events where `include[lv]` is false are skipped
    /// entirely — as if absent from the graph, not merely silent. Used by
    /// `ObjectDoc.compact` to materialize per-object state as of a past
    /// stable point without walking the events strictly after it; `null`
    /// (the normal, hot path) walks everything.
    pub fn replayAll(
        self: *Walker,
        gpa: Allocator,
        first_new: Lv,
        out: *std.ArrayList(Effect),
        include: ?[]const bool,
    ) ReplayError!void {
        const n = self.graph.eventCount();
        try self.reg_pos_of.appendNTimes(gpa, none, n);
        try self.kills_start.appendNTimes(gpa, 0, n);
        try self.kills_len.appendNTimes(gpa, 0, n);
        for (0..n) |lv_usize| {
            const lv: Lv = @intCast(lv_usize);
            if (include) |inc| if (!inc[lv]) continue;
            try seq_walker.movePrepareTo(gpa, self.graph, &self.prep_frontier, self.graph.parentsOf(lv), self);
            try self.apply(gpa, lv, lv >= first_new, out);
            self.prep_frontier.clearRetainingCapacity();
            try self.prep_frontier.append(gpa, lv);
        }
        // Lamport (F3, delta 6) is computed ONLY here, inside
        // `emitStructEffects`, and ONLY when there is at least one
        // structural event — see `lamport`'s doc comment: a struct-free
        // replay (the common case) never allocates or touches it.
        if (self.struct_events.items.len > 0) {
            try self.emitStructEffects(gpa, first_new, out);
        }
    }

    /// `SeqWalker.movePrepareTo`'s per-event callback: advance (`on =
    /// true`) or retreat an already-applied event's prepare effect.
    /// Sequence-shaped ops (list/text ins/del) delegate to the target
    /// object's own `SeqWalker`; map ops are Walker's own (registers
    /// aren't sequences, no `SeqWalker` involvement).
    pub fn toggle(self: *Walker, history: *const Graph, lv: Lv, on: bool) void {
        switch (history.opOf(lv)) {
            .list_ins, .text_ins => self.seqs.getPtr(self.appliedSeqObj(lv)).?.toggleInsert(lv, on),
            .list_del, .text_del => self.seqs.getPtr(self.appliedSeqObj(lv)).?.toggleDelete(lv, on),
            .map_set, .map_del => {
                const info = self.appliedMapSlot(lv);
                const slot = &self.maps.getPtr(info.obj).?.slots.items[info.slot];
                if (history.opOf(lv) == .map_set) {
                    const rp = self.reg_pos_of.items[lv];
                    assert(rp != none);
                    slot.items.items[@intCast(rp)].prep_inserted = on;
                }
                const ks = self.kills_start.items[lv];
                const kl = self.kills_len.items[lv];
                for (self.kills_pool.items[ks..][0..kl]) |victim| {
                    const v = &slot.items.items[victim];
                    if (on) v.prep_overwritten += 1 else v.prep_overwritten -= 1;
                }
            },
            // Structural ops (F3, delta 6) have no per-key/per-sequence
            // prepare-visibility state to retreat/advance — their
            // resolution is a separate, GLOBAL canonical-order pass (see
            // `emitStructEffects`), not relative to any one event's
            // authoring-time frontier.
            .struct_create, .struct_move => {},
        }
    }

    fn appliedSeqObj(self: *const Walker, lv: Lv) Lv {
        const ref: ?ObjId = switch (self.graph.opOf(lv)) {
            .list_ins => |o| o.obj,
            .list_del => |o| o.obj,
            .text_ins => |o| o.obj,
            .text_del => |o| o.obj,
            else => unreachable,
        };
        return self.graph.lvOf(ref.?).?;
    }

    fn appliedMapSlot(self: *const Walker, lv: Lv) struct { obj: Lv, slot: usize } {
        const op = self.graph.opOf(lv);
        const ref: ?ObjId = switch (op) {
            .map_set => |m| m.obj,
            .map_del => |m| m.obj,
            else => unreachable,
        };
        const key: Str = switch (op) {
            .map_set => |m| m.key,
            .map_del => |m| m.key,
            else => unreachable,
        };
        const obj = if (ref) |id| self.graph.lvOf(id).? else root_key;
        const reg = self.maps.getPtr(obj).?;
        for (reg.slots.items, 0..) |s, i| {
            if (std.mem.eql(u8, self.str(s.key), self.str(key))) return .{ .obj = obj, .slot = i };
        }
        unreachable;
    }

    fn getSeqWalker(self: *Walker, gpa: Allocator, obj: Lv) Allocator.Error!*SeqWalker {
        const gop = try self.seqs.getOrPut(gpa, obj);
        if (!gop.found_existing) {
            gop.value_ptr.* = .empty;
            // Seed a freshly-touched object's placeholder items from its
            // compacted base, if it has one (text objects only — see
            // `TextBase`). Must happen before any real item is applied
            // (`SeqWalker.initBase`'s own precondition).
            if (self.text_bases) |bases| {
                if (bases.get(self.graph.idOf(obj))) |base| {
                    try gop.value_ptr.initBase(gpa, base.scalars, seq_walker.base_placeholder_lv);
                }
            }
        }
        return gop.value_ptr;
    }

    fn getSlot(self: *Walker, gpa: Allocator, obj: Lv, key: Str) Allocator.Error!*KeySlot {
        const gop = try self.maps.getOrPut(gpa, obj);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        const reg = gop.value_ptr;
        for (reg.slots.items) |*s| {
            if (std.mem.eql(u8, self.str(s.key), self.str(key))) return s;
        }
        try reg.slots.append(gpa, .{ .key = key });
        return &reg.slots.items[reg.slots.items.len - 1];
    }

    fn apply(self: *Walker, gpa: Allocator, lv: Lv, emit: bool, out: *std.ArrayList(Effect)) ReplayError!void {
        const names: Names = .{ .g = self.graph };
        switch (self.graph.opOf(lv)) {
            .map_set => |m| {
                const obj = try self.resolveObj(m.obj, lv, 0, emit);
                const slot = try self.getSlot(gpa, obj, m.key);
                try self.registerWrite(gpa, lv, obj, slot, m.key, emit, out);
                self.reg_pos_of.items[lv] = @intCast(slot.items.items.len);
                try slot.items.append(gpa, .{
                    .lv = lv,
                    .prep_inserted = true,
                    .prep_overwritten = 0,
                    .effect_active = true,
                });
                if (emit) {
                    const eobj: ?Lv = if (obj == root_key) null else obj;
                    try out.append(gpa, .{ .map_add = .{ .obj = eobj, .key = m.key, .set_lv = lv, .val = m.val } });
                }
            },
            .map_del => |m| {
                const obj = try self.resolveObj(m.obj, lv, 0, emit);
                const slot = try self.getSlot(gpa, obj, m.key);
                try self.registerWrite(gpa, lv, obj, slot, m.key, emit, out);
            },
            .list_ins => |l| {
                const obj = try self.resolveObj(l.obj, lv, 1, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyInsert(gpa, names, lv, l.pos, emit);
                if (emit) {
                    try out.append(gpa, .{ .list_ins = .{ .obj = obj, .index = res.effect_pos, .lv = lv, .val = l.val } });
                }
            },
            .list_del => |l| {
                const obj = try self.resolveObj(l.obj, lv, 1, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyDelete(gpa, lv, l.pos, emit);
                if (emit) {
                    if (res.effect_pos) |index| try out.append(gpa, .{ .list_del = .{ .obj = obj, .index = index } });
                }
            },
            .text_ins => |x| {
                const obj = try self.resolveObj(x.obj, lv, 2, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyInsert(gpa, names, lv, x.pos, emit);
                if (emit) {
                    try out.append(gpa, .{ .text_ins = .{ .obj = obj, .pos = res.effect_pos, .ch = x.ch } });
                }
            },
            .text_del => |x| {
                const obj = try self.resolveObj(x.obj, lv, 2, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyDelete(gpa, lv, x.pos, emit);
                if (emit) {
                    if (res.effect_pos) |pos| try out.append(gpa, .{ .text_del = .{ .obj = obj, .pos = pos } });
                }
            },
            .struct_create => |c| {
                _ = try self.resolveStructRef(c.parent, lv, emit);
                try self.struct_events.append(gpa, lv);
                if (emit) try out.append(gpa, .{ .struct_created = .{ .node = lv } });
            },
            .struct_move => |m| {
                _ = try self.resolveStructNode(m.node, lv, emit);
                _ = try self.resolveStructRef(m.parent, lv, emit);
                try self.struct_events.append(gpa, lv);
            },
        }
    }

    /// Shared map_set/map_del front half: overwrite the prepare-visible
    /// values, recording the kill list and emitting removals.
    fn registerWrite(
        self: *Walker,
        gpa: Allocator,
        lv: Lv,
        obj: Lv,
        slot: *KeySlot,
        key: Str,
        emit: bool,
        out: *std.ArrayList(Effect),
    ) ReplayError!void {
        self.kills_start.items[lv] = @intCast(self.kills_pool.items.len);
        var killed: u32 = 0;
        for (slot.items.items, 0..) |*it, i| {
            if (!it.prepVisible()) continue;
            try self.kills_pool.append(gpa, @intCast(i));
            killed += 1;
            it.prep_overwritten += 1;
            if (it.effect_active) {
                it.effect_active = false;
                if (emit) {
                    const eobj: ?Lv = if (obj == root_key) null else obj;
                    try out.append(gpa, .{ .map_remove = .{ .obj = eobj, .key = key, .set_lv = it.lv } });
                }
            }
        }
        self.kills_len.items[lv] = killed;
    }

    // ── Structural parent-register resolution (F3, delta 6) ────────────
    // `stemma-unification.md` §3 step 5, ported from
    // `structure_sketch.zig`'s `materialize`/`computeLamport`/`CanonCtx`.
    // See landmine 1 (`ObjectDoc.structParent`'s doc comment): this is a
    // GLOBAL, replica-portable total order over every structural write —
    // Lamport per event, ties broken by (agent name, seq) — replayed once
    // with per-write cycle rejection. It is NOT the per-register MV rule
    // `registerWrite`/`mapGet` use above; a tree's parent registers can't
    // be resolved one register at a time, because an edge accepted in one
    // node's register can make a candidate edge in ANOTHER node's
    // register cyclic.
    //
    // Delta from the sketch's `materialize`: the sketch recomputes the
    // WHOLE tree from scratch on every call, into a standalone
    // `Materialized` value a caller queries separately from any replay.
    // Here, resolution is scoped to just the `struct_create`/`struct_move`
    // SUBSET of events (`struct_events`, collected during the ordinary
    // Lv-ordered pass above — cheap when most of the document is
    // map/list/text, unlike the sketch's document which has nothing else),
    // and its OUTPUT is `Effect`s appended to the SAME stream
    // `ObjectDoc.merge`'s existing effect-consuming loop already drives —
    // not a second, disconnected query surface. To decide WHICH nodes
    // actually changed (so `merge`'s `Change` stream stays a real diff,
    // not "every structural node, every merge"), resolution runs twice per
    // replay — once restricted to events strictly before `first_new` (the
    // "before" state: what a prior call already materialized), once over
    // every visited event (the "after" state) — and only nodes whose
    // resolved (key_writer, cycle_broken, conflict_count) tuple differs
    // emit an effect. This is NOT a fully cross-merge-incremental
    // canonical replay (a new event with lower Lamport than
    // already-resolved ones can, in principle, change a distant node's
    // outcome — the before/after diff catches this correctly because it
    // always recomputes the full canonical order over `struct_events`, it
    // just avoids re-touching `map`/`seq` state to do it); true
    // incrementality across separate `Walker` instances is not attempted,
    // matching every OTHER piece of `Walker` state (a fresh `Walker` walks
    // `0..n` on every `replayAll` call already — see the type doc above).

    const ResolvedStruct = struct {
        have_state: bool = false,
        parent: StructTarget = .root,
        order_key: Str = undefined,
        key_writer: Lv = undefined,
        conflict_count: u32 = 0,
        cycle_broken: bool = false,
    };

    /// `ref` (portable `StructRef`) → Walker-local `StructTarget`,
    /// validating existence/causal-order/kind exactly like `resolveObj`
    /// (`untrusted`: true for a new, not-yet-trusted remote event).
    fn resolveStructRef(self: *const Walker, ref: StructRef, at_lv: Lv, untrusted: bool) error{Corrupt}!StructTarget {
        return switch (ref) {
            .root => .root,
            .trash => .trash,
            .node => |id| .{ .node = try self.resolveStructNode(id, at_lv, untrusted) },
        };
    }

    /// Same as `resolveStructRef`'s `.node` case, standalone — used for
    /// `struct_move`'s `node` field, which (unlike `parent`) is never
    /// `.root`/`.trash`.
    fn resolveStructNode(self: *const Walker, id: ObjId, at_lv: Lv, untrusted: bool) error{Corrupt}!Lv {
        const creation = self.graph.lvOf(id) orelse {
            if (untrusted) return error.Corrupt;
            unreachable;
        };
        if (creation >= at_lv) {
            if (untrusted) return error.Corrupt;
            unreachable;
        }
        const ok = switch (self.graph.opOf(creation)) {
            .struct_create => true,
            else => false,
        };
        if (!ok) {
            if (untrusted) return error.Corrupt;
            unreachable;
        }
        return creation;
    }

    /// Same resolution as `resolveStructRef`, but for a ref ALREADY
    /// validated earlier this replay (every `struct_events` member passed
    /// through `resolveStructRef`/`resolveStructNode` in `apply` first) —
    /// used by `resolveStructs`'s canonical-order pass, which re-derives
    /// `parent`/`node` from `graph.opOf` rather than stashing them.
    fn resolveStructRefTrusted(self: *const Walker, ref: StructRef) StructTarget {
        return switch (ref) {
            .root => .root,
            .trash => .trash,
            .node => |id| .{ .node = self.graph.lvOf(id).? },
        };
    }

    /// Would giving `node` the parent `parent` make `node` its own
    /// ancestor, given `states` (already acyclic, by induction — see
    /// `resolveStructs`'s proof note, mirroring
    /// `structure_sketch.zig:wouldCycle`)?
    fn wouldCycleStruct(states: *const std.AutoHashMapUnmanaged(Lv, ResolvedStruct), node: Lv, parent: StructTarget) bool {
        if (parent != .node) return false;
        var cur = parent.node;
        if (cur == node) return true;
        var guard: usize = 0;
        while (true) {
            guard += 1;
            assert(guard <= states.count() + 2); // invariant: chains are finite and acyclic
            const st = states.get(cur) orelse return false;
            if (!st.have_state) return false;
            switch (st.parent) {
                .root, .trash => return false,
                .node => |next| {
                    if (next == node) return true;
                    cur = next;
                },
            }
        }
    }

    /// Resolve every `struct_events` member with `lv < horizon` into final
    /// per-node parent-register state, in GLOBAL Lamport-canonical order
    /// with per-write cycle rejection — see the section doc comment above.
    /// Caller owns the returned map (`freeStructStates`).
    fn resolveStructs(self: *const Walker, gpa: Allocator, horizon: Lv) Allocator.Error!std.AutoHashMapUnmanaged(Lv, ResolvedStruct) {
        var order: std.ArrayList(Lv) = .empty;
        defer order.deinit(gpa);
        for (self.struct_events.items) |lv| {
            if (lv < horizon) try order.append(gpa, lv);
        }
        const SortCtx = struct {
            w: *const Walker,
            fn less(ctx: @This(), a: Lv, b: Lv) bool {
                const la = ctx.w.lamport.items[a];
                const lb = ctx.w.lamport.items[b];
                if (la != lb) return la < lb;
                const ida = ctx.w.graph.idOf(a);
                const idb = ctx.w.graph.idOf(b);
                const name_order = std.mem.order(u8, ctx.w.graph.agentName(ida.agent), ctx.w.graph.agentName(idb.agent));
                if (name_order != .eq) return name_order == .lt;
                return ida.seq < idb.seq;
            }
        };
        std.mem.sort(Lv, order.items, SortCtx{ .w = self }, SortCtx.less);

        var live: std.AutoHashMapUnmanaged(Lv, std.ArrayList(Lv)) = .empty;
        defer {
            var lit = live.valueIterator();
            while (lit.next()) |l| l.deinit(gpa);
            live.deinit(gpa);
        }
        var states: std.AutoHashMapUnmanaged(Lv, ResolvedStruct) = .empty;
        errdefer states.deinit(gpa);

        for (order.items) |lv| {
            const op = self.graph.opOf(lv);
            const node: Lv = switch (op) {
                .struct_create => lv,
                .struct_move => |m| self.graph.lvOf(m.node).?, // validated in `apply`
                else => unreachable,
            };
            const parent: StructTarget = switch (op) {
                .struct_create => |c| self.resolveStructRefTrusted(c.parent),
                .struct_move => |m| self.resolveStructRefTrusted(m.parent),
                else => unreachable,
            };
            const order_key: Str = switch (op) {
                .struct_create => |c| c.order_key,
                .struct_move => |m| m.order_key,
                else => unreachable,
            };

            // Register conflict bookkeeping: keep entries not causally
            // superseded by this write, then add it — identical rule to
            // `registerWrite`'s map-register bookkeeping above, just keyed
            // by node instead of (obj, key).
            const live_gop = try live.getOrPut(gpa, node);
            if (!live_gop.found_existing) live_gop.value_ptr.* = .empty;
            var w: usize = 0;
            for (live_gop.value_ptr.items) |e| {
                const rel = try self.graph.compareFrontiers(gpa, &.{e}, &.{lv});
                if (rel != .ancestor) {
                    live_gop.value_ptr.items[w] = e;
                    w += 1;
                }
            }
            live_gop.value_ptr.items.len = w;
            try live_gop.value_ptr.append(gpa, lv);

            const st_gop = try states.getOrPut(gpa, node);
            if (!st_gop.found_existing) st_gop.value_ptr.* = .{};

            if (op == .struct_move and wouldCycleStruct(&states, node, parent)) continue; // rejected

            st_gop.value_ptr.have_state = true;
            st_gop.value_ptr.parent = parent;
            st_gop.value_ptr.order_key = order_key;
            st_gop.value_ptr.key_writer = lv;
        }

        var sit = states.iterator();
        while (sit.next()) |kv| {
            const l = live.get(kv.key_ptr.*).?;
            kv.value_ptr.conflict_count = @intCast(l.items.len);
            kv.value_ptr.cycle_broken = std.mem.indexOfScalar(Lv, l.items, kv.value_ptr.key_writer) == null;
        }
        return states;
    }

    fn freeStructStates(gpa: Allocator, m: *std.AutoHashMapUnmanaged(Lv, ResolvedStruct)) void {
        m.deinit(gpa);
    }

    /// `lamport[lv] = 1 + max(lamport[parent])` over the WHOLE graph
    /// (every op kind — a structural event's parents can be non-structural
    /// events, since `ObjectDoc` shares ONE `EventGraph` across every op),
    /// a separate pass exactly like `structure_sketch.zig:computeLamport`.
    /// Only ever called from `emitStructEffects`, which only runs when
    /// `struct_events` is non-empty — see `lamport`'s doc comment for why
    /// this is deliberately NOT folded into the main per-event loop above.
    fn computeLamport(self: *Walker, gpa: Allocator) Allocator.Error!void {
        const n = self.graph.eventCount();
        try self.lamport.appendNTimes(gpa, 0, n);
        for (0..n) |lv_usize| {
            const lv: Lv = @intCast(lv_usize);
            var m: u32 = 0;
            for (self.graph.parentsOf(lv)) |p| m = @max(m, self.lamport.items[p] + 1);
            self.lamport.items[lv] = m;
        }
    }

    /// Diff the "before this replay's new events" and "after" resolved
    /// structural state, emitting one `Effect.struct_parent` per node
    /// whose (winner, cycle-break status, conflict count) actually
    /// changed. See the section doc comment above for why this is scoped
    /// to `struct_events` rather than the whole graph, and what
    /// "incremental" does and doesn't mean here.
    fn emitStructEffects(self: *Walker, gpa: Allocator, first_new: Lv, out: *std.ArrayList(Effect)) Allocator.Error!void {
        const n_all: Lv = @intCast(self.graph.eventCount());
        if (first_new >= n_all) return; // nothing new this replay — no-op fast path
        try self.computeLamport(gpa);
        var before = try self.resolveStructs(gpa, first_new);
        defer freeStructStates(gpa, &before);
        var after = try self.resolveStructs(gpa, n_all);
        defer freeStructStates(gpa, &after);

        var it = after.iterator();
        while (it.next()) |kv| {
            const node = kv.key_ptr.*;
            const ns = kv.value_ptr.*;
            if (!ns.have_state) continue; // unreachable in practice: create always accepts
            const changed = if (before.get(node)) |os|
                (!os.have_state or os.key_writer != ns.key_writer or
                    os.cycle_broken != ns.cycle_broken or os.conflict_count != ns.conflict_count)
            else
                true;
            if (!changed) continue;
            try out.append(gpa, .{ .struct_parent = .{
                .node = node,
                .parent = ns.parent,
                .order_key = ns.order_key,
                .key_writer = ns.key_writer,
                .conflict_count = ns.conflict_count,
                .cycle_broken = ns.cycle_broken,
            } });
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
