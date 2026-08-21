//! F3 VALIDATION SKETCH — exploratory, additive, NOT part of the public
//! `stemma` module (see build.zig: it has its own test-only module root and
//! is unreachable from root.zig). Purpose: validate, with running code and
//! property tests, the decision recorded in app/weft/doc/north-star-plan.md
//! §5 F3 — "parent-register + fractional order keys for structural
//! children" — BEFORE any weft client (a structural editor, dired-over-a-
//! graph, etc.) depends on it. If this sketch had found the representation
//! unworkable, F3 would need to be revisited; §7 of that plan names this
//! risk explicitly ("F3 could be wrong in a way only a structural editor
//! reveals"). This file is that stress test, done early and cheaply.
//!
//! Consumes only `causal.zig`'s `EventGraph` substrate (a sibling file, not
//! `objects_state.zig`/`ObjectDoc.zig`) — it does not modify, import, or
//! depend on any production doc-core module, and none of them import it.
//!
//! ## The model
//!
//! A node's identity is the event that created it (`EventId`, mirroring
//! `ObjId` in objects_state.zig). Two permanent, ownerless roots exist
//! without a creation event: `.root` (the visible tree) and `.trash` (where
//! deleted subtrees go — see "Deletion" below). Every real node's placement
//! is governed by a single per-node PARENT REGISTER: a multi-value register
//! (mirroring ObjectDoc's map-register semantics — concurrent writes all
//! survive as a conflict set; reads pick a deterministic winner) whose
//! value is `(parent: NodeRef, order_key: bytes)`.
//!
//! `create` and `move` are the only ops, and they are symmetric: `create`
//! both allocates a node AND performs its first parent-register write in
//! one event; `move` performs a subsequent write against an existing node.
//! This is the sketch's first confirmation of F3's rationale: an identity-
//! preserving move genuinely is exactly one register write, no different in
//! shape from a second `create`.
//!
//! ## Order keys: byte-string fractional midpoints, not rational pairs
//!
//! Chosen representation: a growable byte string (base-256 digit string,
//! most-significant digit first) compared lexicographically, with the
//! authoring event's `EventId` as a final tiebreak so no two keys ever
//! compare equal even when two replicas independently compute the identical
//! digit midpoint — common, e.g. two replicas concurrently appending both
//! compute `between(last, null)` against the same bound. `between(a, b)`
//! finds a value strictly inside the open interval by walking digit
//! positions until room opens up (`db - da >= 2`), extending the string by
//! one digit when it doesn't (see `between` below for the algorithm); it
//! REQUIRES `a` strictly precede `b` (asserted), so a caller that finds
//! tied digit neighbors — as the property-test generator's `randomGapKey`
//! does — must widen past the tied run to a genuinely distinct bound (or
//! `null`) before calling it; `between()` itself has no use for the
//! tiebreak, only sibling comparison (`Materialized.childrenOf`) does.
//!
//! Rejected alternative: rational pairs (arbitrary-precision numerator/
//! denominator, e.g. Logoot's original scheme or naive "average the two
//! floats"). Byte-strings were chosen because (a) comparison is a single
//! `mem.order` call with no arithmetic, (b) there is no periodic
//! renormalization step to get wrong — growth is local to the one gap being
//! subdivided, never a whole-sequence rescale, and (c) it wire-encodes as
//! plain bytes with no int-width ceiling to hit. The cost, examined
//! honestly below, is unbounded-worst-case key length — which rational
//! pairs do not actually avoid either (an arbitrary-precision numerator
//! under repeated bisection grows exactly the same number of BITS; the
//! bound below is representation-independent, not a byte-string weakness).
//!
//! ### The dense-insertion pathological case, handled honestly
//!
//! `between()` picks the true midpoint of the available digit range at each
//! position, which is the right choice for RANDOM insertion patterns (gives
//! ~O(log N) expected depth for N insertions scattered across a sequence —
//! `"random insertions keep keys short"` below measures this). But for the
//! ADVERSARIAL pattern this sketch actually measures — N insertions always
//! immediately against the SAME shrinking boundary (always-prepend,
//! always-append, or "always nudge this one item further" drag-reordering,
//! i.e. `between(fixed_side, moving_side)` called N times with `moving_side`
//! replaced by the previous result each time) — repeated bisection of one
//! shrinking interval is provably Θ(N) in digit COUNT, not O(log N): each
//! byte of range (256 values) only absorbs ~8 bisections (256→128→64→…→1)
//! before it's exhausted and a new digit is appended, so N adversarial
//! insertions cost ~N/8 bytes of growth. `"pathological same-gap insertion
//! grows keys linearly, not logarithmically"` below measures and asserts
//! this honestly rather than claiming a log bound this algorithm cannot
//! deliver for that pattern.
//!
//! Precision matters on WHOSE floor this is, though (corrected after
//! review — an earlier draft overclaimed it as universal). The append/
//! prepend-into-a-shrinking-bound growth measured above is a property of
//! THIS scheme's unconditional midpoint bisection, applied uniformly to
//! open (infinite) and bounded sides alike — it is NOT unavoidable for
//! every dense fractional key scheme. LSEQ specifically escapes it for the
//! append/prepend case by allocating from a boundary with room to keep
//! growing outward at the SAME depth (a different, side-aware strategy),
//! rather than always bisecting toward the midpoint — real production
//! fractional-indexing libraries do this too. What genuinely IS a
//! universal, unavoidable floor — escaped by nothing without relabeling —
//! is inserting arbitrarily many times strictly between two FIXED,
//! already-adjacent keys that never themselves move: since both endpoints
//! are pinned, distinguishing N new points nested inside that one fixed
//! gap requires Θ(N) bits of resolution in the worst case, full stop, for
//! ANY dense immutable-label scheme (rational, Logoot, LSEQ, or this one).
//! This sketch measures the scheme-specific case, not the universal one
//! — a real implementation may want a boundary-aware `between()` to push
//! the append/prepend cost down, but no encoding trick removes the fixed-
//! gap floor. Either way the production implication is the same, named in
//! FINDINGS: a real implementation needs periodic sibling-key REBALANCING
//! (renumber a fragmented run during compaction) as the actual fix for
//! the cases that DO hit their floor — a cleverer encoding narrows which
//! patterns need it, but cannot eliminate the need.
//!
//! ## Move, and the cycle rule
//!
//! Concurrent moves of the SAME node are the register's ordinary MV-conflict
//! case and need no special handling: both writes survive as conflict-set
//! members (see `Materialized.conflictCountOf`), and the deterministic
//! winner is computed the same way for every node uniformly (below).
//!
//! Concurrent moves of DIFFERENT nodes that are individually acyclic but
//! jointly cyclic (A moves under B, concurrently B moves under A) are the
//! hard case, and are NOT a per-register problem — resolving each node's
//! register independently (as ObjectDoc does for map keys) cannot see the
//! cross-node cycle. The rule implemented here (`materialize` below),
//! Kleppmann-style:
//!
//! 1. Compute a REPLICA-PORTABLE total order over every write in the graph:
//!    a Lamport timestamp per event (`1 + max(lamport(parent))`, 0 for
//!    genesis), sorted by `(lamport, agent_name, seq)`. This is a valid
//!    linearization of the causal DAG (parents always sort before
//!    children, since a parent's Lamport value is always strictly smaller)
//!    with concurrent events broken by portable identity — NOT by local
//!    `Lv`, which is explicitly documented in causal.zig as replica-local
//!    and unstable across replicas. Two replicas that have seen the same
//!    event set always compute the identical order.
//! 2. Replay writes in that order, maintaining a live `effective_parent`
//!    map. Each write is applied (overwriting the node's effective parent)
//!    UNLESS applying it would make the node its own ancestor, in which
//!    case it is deterministically SKIPPED — the node keeps whatever its
//!    last accepted write set. The graph being built is acyclic by
//!    induction (empty is acyclic; each accepted edge is checked against
//!    the current, already-acyclic graph before being added), so the
//!    result is unconditionally cycle-free, and identical on every replica
//!    because the order and the rule are both portable and deterministic.
//!
//! FINDING (surprising, recorded honestly): this means the register's
//! "deterministic winner" is NOT always the same rule ObjectDoc's map
//! register uses (greatest `(agent_name, seq)` among the causally-maximal
//! antichain). It is the last write in canonical order whose application
//! does not cycle — which differs from ObjectDoc's rule in two ways, one
//! merely different and one that needed correcting after review.
//!
//! First (merely different, benign): among writes that all DO apply
//! cleanly, "last in Lamport order" can differ from "greatest
//! `(agent_name, seq)`" when concurrent writers sit at different causal
//! depths (a deep chain can outrank a shallow one alphabetically
//! "losing") — but it is still always a causally-maximal conflict-set
//! member in that case, just not necessarily the SAME member ObjectDoc's
//! simpler per-key rule would pick.
//!
//! Second (a real correction — an earlier draft of this doc claimed the
//! winner is ALWAYS a conflict-set member; that is false, and a reviewer's
//! counterexample caught it): when the causally-dominant write to a
//! register would cycle, it is skipped, and the effective parent falls
//! back to whichever EARLIER write last applied cleanly. But the
//! conflict-set bookkeeping (`conflict_live`) removes an entry from a
//! register's antichain as soon as ANY causally-later write to that
//! register exists — including one that is itself later rejected for
//! cycling. So when every causally-later write to a register would cycle,
//! the materialized winner is an earlier, already-superseded write (in
//! the limit, the node's own `create`) that sits OUTSIDE the reported
//! conflict set entirely. This is still fully deterministic and
//! convergent — every replica computes the same canonical order and the
//! same sequence of accept/reject decisions — it just cannot be described
//! as "pick the antichain's greatest member" the way ObjectDoc's map
//! register can. Pinned exactly by
//! `"a rejected write can leave an earlier, already-superseded write as
//! the effective parent — outside the reported conflict set"` below; see
//! the corrected proof note on `materialize`.
//!
//! The underlying reason both differences exist: ObjectDoc's map keys
//! never need cross-key arbitration, so it can resolve each register in
//! isolation; a tree's parent registers cannot, because an edge in one
//! register can make an edge in another register cyclic, and cycle
//! rejection can strand an old write as the winner. This is a genuine,
//! structural difference from the map case that F3's rationale ("the same
//! MV-register semantics ObjectDoc already has for maps") understates —
//! worth flagging for whoever writes the real implementation.
//!
//! ## Deletion
//!
//! `delete(node)` is sugar for `move(node, .trash, "")` — NOT a distinct op,
//! and NOT recursive. A deleted node's children are untouched: their own
//! parent registers still point at it. Materializing the tree from `.root`
//! therefore stops seeing them (their ancestor chain hits the tombstone
//! before reaching `.root`), but they are not destroyed, renumbered, or
//! reparented — moving the deleted node back out of `.trash` resurrects the
//! entire subtree exactly as it was, INCLUDING any structural edits made to
//! it while it was hidden (a peer could keep editing "under" a node the
//! local replica had deleted; on sync, that work is not lost, just hidden
//! until someone restores the parent). This was a deliberate choice among
//! three: (a) promote children to the deleted node's own parent — rejected,
//! it silently rewrites structure no one asked for and is NOT what most
//! apps mean by delete; (b) hard-delete the whole subtree recursively —
//! rejected, it requires visiting potentially unbounded descendants for a
//! single op and is irreversible, which a CRDT (no true deletes, only
//! tombstones) should avoid without a strong reason; (c) tombstone-as-
//! trash-reparent — the one implemented, O(1) per delete regardless of
//! subtree size, trivially reversible, and reuses the exact same
//! parent-register + cycle machinery with zero special-casing (`.trash` is
//! just another `NodeRef`). `"delete hides a subtree without destroying
//! it; undelete restores it intact"` below is the property test.
//!
//! ## FINDINGS
//!
//! Parent-register + fractional order keys HOLDS UP as a representation:
//! move is genuinely one write; concurrent-move-of-one-node is genuinely
//! ObjectDoc's existing MV-register story with no new mechanism; deletion
//! composes for free out of "trash is just another parent," needing no
//! separate tombstone bookkeeping or subtree walk. Convergence, acyclicity,
//! reachability, and order-stability all held across the property-test
//! campaign described in the test at the bottom of this file (see that
//! test's doc comment for run counts and what broke during development).
//!
//! What surprised: (1) the winner-selection rule is NOT simply "reuse
//! ObjectDoc's map-register rule," as F3's rationale implies — cross-node
//! cycle-breaking forces a single GLOBAL total order (this sketch uses
//! Lamport timestamps) that per-register resolution doesn't need, which is
//! a genuinely new mechanism, not a reused one; a sharper version of the
//! same surprise is that cycle-rejection can strand an EARLIER write as
//! the winner, outside the reported conflict set (see the FINDING above
//! and its pinned counterexample test — this was wrong in an earlier draft
//! of this doc until a review caught it). (2) This scheme's order-key
//! growth is linear, not logarithmic, under append/prepend-style
//! same-gap insertion — worth naming since "fractional keys" is sometimes
//! assumed to be a free O(log N) lunch; that specific growth rate is a
//! property of this scheme's unconditional midpoint bisection (schemes
//! like LSEQ avoid it for append/prepend), though the deeper floor —
//! inserting forever strictly between two keys that never move — really
//! is universal, for anyone, in any encoding (see the order-key section
//! above for the precise distinction).
//!
//! What the real (non-sketch) implementation needs that this elides:
//! - COMPACTION interaction: nothing here ever prunes graph history or
//!   rebalances a fragmented sibling run. The linear-growth pathological
//!   case (above) is exactly the scenario production compaction must
//!   periodically fix (renumber a node's children with fresh, evenly
//!   spaced keys) — this sketch only measures the problem, not the fix.
//! - WIRE ENCODING: sync here is in-process event translation
//!   (`syncInto` below), analogous to what ObjectDoc's `Decoder`/
//!   `encodeEvents` do over real bytes, but never touches a byte buffer.
//!   A real implementation needs the actual wire format (and the same
//!   hostile-input hardening object_tests.zig exercises for ObjectDoc).
//! - IN-NODE ANCHORS: nothing here addresses referencing a position INSIDE
//!   a structural node's own content (e.g. an anchor into one child among
//!   many) — F3 is purely about the parent/child/order skeleton; content
//!   identity anchors are `stemma delta 3` in the plan, a separate concern.
//! - Materialization here is recomputed from scratch on every call
//!   (`materialize`) — fine for a sketch and for property tests at this
//!   scale, but the real implementation needs the incremental prepare/
//!   effect discipline `objects_state.zig`'s `Walker` already has for maps
//!   and sequences, extended to the parent register + Lamport ordering.
//!
//! No evidence surfaced AGAINST the F3 decision itself (parent-register
//! over lists for structural children). The one real caveat is the
//! winner-rule/Lamport-order point above: it's an added mechanism, not a
//! reused one, and should be budgeted as such rather than assumed free.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
pub const Lv = causal.Lv;

fn eqId(a: EventId, b: EventId) bool {
    return a.agent == b.agent and a.seq == b.seq;
}

/// String reference into a `Doc`'s append-only key arena (mirrors
/// objects_state.zig's `Str`).
pub const Str = struct { start: u32, len: u32 };

/// Where a node's parent register can point: the two permanent roots, or
/// another real node by identity.
pub const NodeRef = union(enum) {
    root,
    trash,
    node: EventId,

    fn eql(a: NodeRef, b: NodeRef) bool {
        return switch (a) {
            .root => b == .root,
            .trash => b == .trash,
            .node => |na| switch (b) {
                .node => |nb| eqId(na, nb),
                else => false,
            },
        };
    }
};

pub const StructureOp = union(enum) {
    /// Allocates a new node (identity = this event's own id) and performs
    /// its first parent-register write in the same event.
    create: struct { parent: NodeRef, key_digits: Str },
    /// A subsequent parent-register write against an existing node.
    move: struct { node: EventId, parent: NodeRef, key_digits: Str },
};

pub const Graph = causal.EventGraph(StructureOp);

// ── Order keys ─────────────────────────────────────────────────────────

/// Byte-string fractional midpoint strictly between `a` and `b` (`null` a =
/// -infinity, `null` b = +infinity). Caller owns the returned slice. See
/// the module doc for the growth-bound discussion.
pub fn between(gpa: Allocator, a: ?[]const u8, b: ?[]const u8) Allocator.Error![]u8 {
    // Structural precondition: `a` strictly precedes `b` whenever both are
    // bounded (`null` stands for -infinity/+infinity respectively, always
    // satisfied). Callers are expected to pass adjacent, already-sorted
    // neighbors — this is not a caller-facing error case, it's a caller
    // bug, so it asserts rather than returning an error.
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

// ── The doc ────────────────────────────────────────────────────────────

pub const Doc = struct {
    history: Graph = .empty,
    agent: ?AgentId = null,
    /// Append-only arena for key digit bytes (Str refs point here).
    keys: std.ArrayList(u8) = .empty,

    pub const empty: Doc = .{};

    pub fn deinit(self: *Doc, gpa: Allocator) void {
        self.history.deinit(gpa);
        self.keys.deinit(gpa);
        self.* = .{};
    }

    pub fn setAgent(self: *Doc, gpa: Allocator, name: []const u8) Allocator.Error!void {
        self.agent = try self.history.registerAgent(gpa, name);
    }

    fn internKey(self: *Doc, gpa: Allocator, digits: []const u8) Allocator.Error!Str {
        const start: u32 = @intCast(self.keys.items.len);
        try self.keys.appendSlice(gpa, digits);
        return .{ .start = start, .len = @intCast(digits.len) };
    }

    pub fn keyBytes(self: *const Doc, s: Str) []const u8 {
        return self.keys.items[s.start..][0..s.len];
    }

    /// Create a new node under `parent` with the given order-key digits
    /// (see `between`). Returns the new node's identity.
    pub fn create(self: *Doc, gpa: Allocator, parent: NodeRef, key_digits: []const u8) Allocator.Error!EventId {
        const agent = self.agent.?;
        const ks = try self.internKey(gpa, key_digits);
        const lv = try self.history.addLocal(gpa, agent, .{ .create = .{ .parent = parent, .key_digits = ks } });
        return self.history.idOf(lv);
    }

    /// Identity-preserving move: one parent-register write.
    pub fn move(self: *Doc, gpa: Allocator, node: EventId, parent: NodeRef, key_digits: []const u8) Allocator.Error!void {
        const agent = self.agent.?;
        const ks = try self.internKey(gpa, key_digits);
        _ = try self.history.addLocal(gpa, agent, .{ .move = .{ .node = node, .parent = parent, .key_digits = ks } });
    }

    /// Sugar: move to `.trash`. Not recursive — see module doc "Deletion".
    pub fn delete(self: *Doc, gpa: Allocator, node: EventId) Allocator.Error!void {
        try self.move(gpa, node, .trash, &.{});
    }

    /// In-process "gossip": pull every event `from` has that `self` lacks,
    /// translating replica-local `AgentId`s via portable agent NAMES
    /// (exactly what ObjectDoc's wire decode does, minus the byte
    /// encoding — see FINDINGS on elided wire encoding).
    pub fn syncFrom(self: *Doc, gpa: Allocator, from: *const Doc) Allocator.Error!void {
        // `self.history.frontier.items` is LOCAL to `self` (`Lv` is
        // replica-local, per causal.zig) — it cannot be passed directly as
        // `from`'s `missingFrom` argument, which indexes `from`'s own
        // event array. Translate it into `from`'s local Lv space first via
        // portable (agent name, seq), exactly as `core.decodeVersion` does
        // for a real version token; entries `from` has never seen are
        // simply omitted (lenient, matching decodeVersion's non-strict
        // mode) — `from` doesn't need to know about them to compute what
        // `self` is missing.
        var known_to_from: std.ArrayList(Lv) = .empty;
        defer known_to_from.deinit(gpa);
        for (self.history.frontier.items) |lv| {
            const id = self.history.idOf(lv);
            const name = self.history.agentName(id.agent);
            if (from.history.findAgent(name)) |aid| {
                if (from.history.lvOf(.{ .agent = aid, .seq = id.seq })) |flv| {
                    try known_to_from.append(gpa, flv);
                }
            }
        }
        var missing = try from.history.missingFrom(gpa, known_to_from.items);
        defer missing.deinit(gpa);
        for (missing.items) |lv| {
            const src_id = from.history.idOf(lv);
            const name = from.history.agentName(src_id.agent);
            const dst_agent = try self.history.registerAgent(gpa, name);
            const dst_id: EventId = .{ .agent = dst_agent, .seq = src_id.seq };
            if (self.history.isKnown(dst_id)) continue;

            var parent_lvs: std.ArrayList(Lv) = .empty;
            defer parent_lvs.deinit(gpa);
            for (from.history.parentsOf(lv)) |p| {
                const p_id = from.history.idOf(p);
                const p_name = from.history.agentName(p_id.agent);
                const p_dst_agent = try self.history.registerAgent(gpa, p_name);
                try parent_lvs.append(gpa, self.history.lvOf(.{ .agent = p_dst_agent, .seq = p_id.seq }).?);
            }
            const op = try self.translateOp(gpa, from, from.history.opOf(lv));
            _ = try self.history.add(gpa, dst_id, parent_lvs.items, op);
        }
    }

    fn translateRef(self: *Doc, gpa: Allocator, from: *const Doc, r: NodeRef) Allocator.Error!NodeRef {
        return switch (r) {
            .root, .trash => r,
            .node => |id| blk: {
                const name = from.history.agentName(id.agent);
                const dst_agent = try self.history.registerAgent(gpa, name);
                break :blk .{ .node = .{ .agent = dst_agent, .seq = id.seq } };
            },
        };
    }

    fn translateOp(self: *Doc, gpa: Allocator, from: *const Doc, op: StructureOp) Allocator.Error!StructureOp {
        return switch (op) {
            .create => |c| .{ .create = .{
                .parent = try self.translateRef(gpa, from, c.parent),
                .key_digits = try self.internKey(gpa, from.keyBytes(c.key_digits)),
            } },
            .move => |m| .{ .move = .{
                .node = (try self.translateRef(gpa, from, .{ .node = m.node })).node,
                .parent = try self.translateRef(gpa, from, m.parent),
                .key_digits = try self.internKey(gpa, from.keyBytes(m.key_digits)),
            } },
        };
    }
};

// ── Materialization: canonical order, cycle-safe replay, the tree ───────

fn computeLamport(gpa: Allocator, graph: *const Graph) Allocator.Error![]u32 {
    const n = graph.eventCount();
    const lamport = try gpa.alloc(u32, n);
    for (0..n) |i| {
        const lv: Lv = @intCast(i);
        var m: u32 = 0;
        for (graph.parentsOf(lv)) |p| m = @max(m, lamport[p] + 1);
        lamport[lv] = m;
    }
    return lamport;
}

const CanonCtx = struct {
    graph: *const Graph,
    lamport: []const u32,

    fn less(self: @This(), a: Lv, b: Lv) bool {
        if (self.lamport[a] != self.lamport[b]) return self.lamport[a] < self.lamport[b];
        const ida = self.graph.idOf(a);
        const idb = self.graph.idOf(b);
        const name_order = std.mem.order(u8, self.graph.agentName(ida.agent), self.graph.agentName(idb.agent));
        if (name_order != .eq) return name_order == .lt;
        return ida.seq < idb.seq;
    }
};

const NodeState = struct { parent: NodeRef, key_digits: Str, key_writer: EventId };

pub const Materialized = struct {
    doc: *const Doc,
    order: []Lv,
    lamport: []u32,
    states: std.AutoHashMapUnmanaged(EventId, NodeState) = .empty,
    /// Writes skipped because applying them would have created a cycle
    /// (for introspection/property assertions, not consumed elsewhere).
    rejected: std.ArrayList(Lv) = .empty,
    /// Per-node causally-maximal antichain of writes to its register
    /// (mirrors ObjectDoc's map conflict set) — informational; see
    /// FINDINGS on how this can diverge from the materialized winner.
    conflict_live: std.AutoHashMapUnmanaged(EventId, std.ArrayList(Lv)) = .empty,

    pub fn deinit(self: *Materialized, gpa: Allocator) void {
        gpa.free(self.order);
        gpa.free(self.lamport);
        self.states.deinit(gpa);
        self.rejected.deinit(gpa);
        var it = self.conflict_live.valueIterator();
        while (it.next()) |l| l.deinit(gpa);
        self.conflict_live.deinit(gpa);
    }

    pub fn conflictCountOf(self: *const Materialized, node: EventId) usize {
        return if (self.conflict_live.get(node)) |l| l.items.len else 0;
    }

    pub fn parentOf(self: *const Materialized, node: EventId) ?NodeRef {
        return if (self.states.get(node)) |s| s.parent else null;
    }

    /// Children of `parent`, sorted by order key (digits, then the
    /// authoring event as a portable tiebreak). Caller owns.
    pub fn childrenOf(self: *const Materialized, gpa: Allocator, parent: NodeRef) Allocator.Error![]EventId {
        var out: std.ArrayList(EventId) = .empty;
        errdefer out.deinit(gpa);
        var it = self.states.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.parent.eql(parent)) try out.append(gpa, kv.key_ptr.*);
        }
        const SortCtx = struct {
            m: *const Materialized,
            fn less(ctx: @This(), a: EventId, b: EventId) bool {
                const sa = ctx.m.states.get(a).?;
                const sb = ctx.m.states.get(b).?;
                const order = std.mem.order(u8, ctx.m.doc.keyBytes(sa.key_digits), ctx.m.doc.keyBytes(sb.key_digits));
                if (order != .eq) return order == .lt;
                const na = ctx.m.doc.history.agentName(sa.key_writer.agent);
                const nb = ctx.m.doc.history.agentName(sb.key_writer.agent);
                const name_order = std.mem.order(u8, na, nb);
                if (name_order != .eq) return name_order == .lt;
                return sa.key_writer.seq < sb.key_writer.seq;
            }
        };
        std.mem.sort(EventId, out.items, SortCtx{ .m = self }, SortCtx.less);
        return out.toOwnedSlice(gpa);
    }

    /// Portable canonical dump (agent NAME + seq, never local ids) for
    /// convergence comparison across replicas.
    pub fn dump(self: *const Materialized, gpa: Allocator) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, "ROOT\n");
        try self.dumpChildren(gpa, &out, .root, 1);
        try out.appendSlice(gpa, "TRASH\n");
        try self.dumpChildren(gpa, &out, .trash, 1);
        return out.toOwnedSlice(gpa);
    }

    fn dumpChildren(self: *const Materialized, gpa: Allocator, out: *std.ArrayList(u8), parent: NodeRef, depth: usize) Allocator.Error!void {
        const kids = try self.childrenOf(gpa, parent);
        defer gpa.free(kids);
        for (kids) |k| {
            for (0..depth) |_| try out.appendSlice(gpa, "  ");
            try out.appendSlice(gpa, self.doc.history.agentName(k.agent));
            try out.print(gpa, "#{d}\n", .{k.seq});
            try self.dumpChildren(gpa, out, .{ .node = k }, depth + 1);
        }
    }

    /// Independent oracle: walk from both roots, asserting every tracked
    /// node is reached exactly once (no orphans, no cycles) — does not
    /// trust `materialize`'s own bookkeeping, re-derives it from
    /// `childrenOf`.
    pub fn verifyForest(self: *const Materialized, gpa: Allocator) !void {
        var visited: std.AutoHashMapUnmanaged(EventId, void) = .empty;
        defer visited.deinit(gpa);
        try self.walkCount(gpa, .root, &visited);
        try self.walkCount(gpa, .trash, &visited);
        if (visited.count() != self.states.count()) return error.OrphanedNodes;
    }

    fn walkCount(self: *const Materialized, gpa: Allocator, parent: NodeRef, visited: *std.AutoHashMapUnmanaged(EventId, void)) !void {
        const kids = try self.childrenOf(gpa, parent);
        defer gpa.free(kids);
        for (kids) |k| {
            const gop = try visited.getOrPut(gpa, k);
            if (gop.found_existing) return error.CycleDetected;
            try self.walkCount(gpa, .{ .node = k }, visited);
        }
    }
};

/// Portable node handle for the property-test generator: (creator agent
/// name, seq), resolved to a replica-local `EventId` lazily, only for
/// replicas that have actually synced the creating event.
const NodeHandle = struct { name: []const u8, seq: u64 };

/// Would giving `node` the parent `parent` make `node` its own ancestor,
/// given the CURRENT (already acyclic, by induction) state? `.root`/
/// `.trash` parents can never cycle.
fn wouldCycle(states: *const std.AutoHashMapUnmanaged(EventId, NodeState), node: EventId, parent: NodeRef) bool {
    if (parent != .node) return false;
    var cur = parent.node;
    if (eqId(cur, node)) return true;
    var guard: usize = 0;
    while (true) {
        guard += 1;
        assert(guard <= states.count() + 2); // invariant: chains are always finite and acyclic
        const st = states.get(cur) orelse return false;
        switch (st.parent) {
            .root, .trash => return false,
            .node => |next| {
                if (eqId(next, node)) return true;
                cur = next;
            },
        }
    }
}

/// Recompute the whole tree from scratch: canonical (Lamport) order, then a
/// single cycle-safe pass. See the module doc for the algorithm and why it
/// needs to be a single global order rather than per-register resolution.
///
/// PROOF that the result is always acyclic: induction on the replay. The
/// empty state is acyclic. Each write is applied only if, in the CURRENT
/// (by hypothesis acyclic) state, the new edge does not create a path back
/// to `node` — i.e. `wouldCycle` is false — so the state after applying it
/// is acyclic too. Rejected writes leave the (acyclic) state unchanged.
///
/// WHAT the materialized winner is (corrected after review — an earlier
/// version of this note claimed the winner is always a causally-maximal
/// `conflict_live` member; that is FALSE, see the counterexample test
/// `"a rejected write can leave an earlier, already-superseded write as
/// the effective parent — outside the reported conflict set"` below). The
/// effective parent recorded in `states` is exactly: the last write to
/// that node's register, in canonical order, whose application did not
/// cycle. When the causally-dominant write to a register applies cleanly,
/// this coincides with "the antichain's greatest member" (the argument:
/// canonical order is topological, so a write causally superseding W must
/// appear after W; if that superseding write also applies cleanly, IT
/// becomes the new winner, and W is correctly both dead in `states` and
/// correctly absent from `conflict_live`). But `conflict_live` kills W's
/// membership as soon as ANY causally-later write to the register exists —
/// including one that itself goes on to be REJECTED for cycling. In that
/// case `states` still holds W (nothing superseded it in `states`, because
/// the only candidate superseder was rejected), while `conflict_live` has
/// already dropped W from the antichain it reports. So in the limiting
/// case where every causally-later write to a register would cycle, the
/// materialized winner is a write that `conflictCountOf`/`conflict_live`
/// no longer lists — outside the reported conflict set, not a bug in
/// `states`, but a real gap between what `conflict_live` reports and what
/// actually won. Both are still fully deterministic and identical across
/// replicas (canonical order and the accept/reject sequence are both
/// portable), so convergence and acyclicity are unaffected — only the
/// "winner is always a legitimate antichain member" claim was wrong.
pub fn materialize(doc: *const Doc, gpa: Allocator) Allocator.Error!Materialized {
    const graph = &doc.history;
    const n = graph.eventCount();
    const order = try gpa.alloc(Lv, n);
    errdefer gpa.free(order);
    for (0..n) |i| order[i] = @intCast(i);
    const lamport = try computeLamport(gpa, graph);
    errdefer gpa.free(lamport);
    std.mem.sort(Lv, order, CanonCtx{ .graph = graph, .lamport = lamport }, CanonCtx.less);

    var m: Materialized = .{ .doc = doc, .order = order, .lamport = lamport };
    errdefer m.deinit(gpa);

    for (order) |lv| {
        const id = graph.idOf(lv);
        const op = graph.opOf(lv);
        const node: EventId = switch (op) {
            .create => id,
            .move => |mv| mv.node,
        };
        const parent: NodeRef = switch (op) {
            .create => |c| c.parent,
            .move => |mv| mv.parent,
        };
        const key_digits: Str = switch (op) {
            .create => |c| c.key_digits,
            .move => |mv| mv.key_digits,
        };

        // Register conflict bookkeeping: keep entries not causally
        // superseded by this write, then add it.
        const live_gop = try m.conflict_live.getOrPut(gpa, node);
        if (!live_gop.found_existing) live_gop.value_ptr.* = .empty;
        var w: usize = 0;
        for (live_gop.value_ptr.items) |e| {
            const rel = try graph.compareFrontiers(gpa, &.{e}, &.{lv});
            if (rel != .ancestor) {
                live_gop.value_ptr.items[w] = e;
                w += 1;
            }
        }
        live_gop.value_ptr.items.len = w;
        try live_gop.value_ptr.append(gpa, lv);

        if (op == .move and wouldCycle(&m.states, node, parent)) {
            try m.rejected.append(gpa, lv);
            continue;
        }
        try m.states.put(gpa, node, .{ .parent = parent, .key_digits = key_digits, .key_writer = id });
    }
    return m;
}

// ── Tests ─────────────────────────────────────────────────────────────

const t = std.testing;

test "create + move: reading back the tree, one register write per move" {
    const gpa = t.allocator;
    var d: Doc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const mid = try between(gpa, null, null);
    defer gpa.free(mid);
    const a = try d.create(gpa, .root, mid);
    const before_count = d.history.eventCount();

    const k2 = try between(gpa, mid, null);
    defer gpa.free(k2);
    const b = try d.create(gpa, .root, k2);

    var m1 = try materialize(&d, gpa);
    defer m1.deinit(gpa);
    const roots1 = try m1.childrenOf(gpa, .root);
    defer gpa.free(roots1);
    try t.expectEqual(@as(usize, 2), roots1.len);
    try t.expect(eqId(roots1[0], a));
    try t.expect(eqId(roots1[1], b));

    // Move b under a: exactly one new event, still the register write.
    const k3 = try between(gpa, null, null);
    defer gpa.free(k3);
    try d.move(gpa, b, .{ .node = a }, k3);
    try t.expectEqual(before_count + 2, d.history.eventCount());

    var m2 = try materialize(&d, gpa);
    defer m2.deinit(gpa);
    const roots2 = try m2.childrenOf(gpa, .root);
    defer gpa.free(roots2);
    try t.expectEqual(@as(usize, 1), roots2.len);
    const a_kids = try m2.childrenOf(gpa, .{ .node = a });
    defer gpa.free(a_kids);
    try t.expectEqual(@as(usize, 1), a_kids.len);
    try t.expect(eqId(a_kids[0], b));
    try m2.verifyForest(gpa);
}

test "concurrent move-move of the same node: both survive as conflicts, deterministic winner, both replicas converge" {
    const gpa = t.allocator;
    var alice: Doc = .empty;
    defer alice.deinit(gpa);
    var bob: Doc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const k0 = try between(gpa, null, null);
    defer gpa.free(k0);
    const shared = try alice.create(gpa, .root, k0);
    const k1 = try between(gpa, k0, null);
    defer gpa.free(k1);
    const parent_a = try alice.create(gpa, .root, k1);
    const k2 = try between(gpa, k1, null);
    defer gpa.free(k2);
    const parent_b = try alice.create(gpa, .root, k2);
    try bob.syncFrom(gpa, &alice);

    const bob_shared: EventId = .{ .agent = bob.history.findAgent("alice").?, .seq = shared.seq };
    const bob_parent_b: EventId = .{ .agent = bob.history.findAgent("alice").?, .seq = parent_b.seq };

    // Concurrent moves of the SAME node (`shared`) to two different parents.
    const ka = try between(gpa, null, null);
    defer gpa.free(ka);
    try alice.move(gpa, shared, .{ .node = parent_a }, ka);
    const kb = try between(gpa, null, null);
    defer gpa.free(kb);
    try bob.move(gpa, bob_shared, .{ .node = bob_parent_b }, kb);

    try bob.syncFrom(gpa, &alice);
    try alice.syncFrom(gpa, &bob);

    var m_alice = try materialize(&alice, gpa);
    defer m_alice.deinit(gpa);
    var m_bob = try materialize(&bob, gpa);
    defer m_bob.deinit(gpa);

    const dump_a = try m_alice.dump(gpa);
    defer gpa.free(dump_a);
    const dump_b = try m_bob.dump(gpa);
    defer gpa.free(dump_b);
    try t.expectEqualStrings(dump_a, dump_b);

    // Honest MV: 2 concurrent writes to `shared`'s register both survive.
    try t.expectEqual(@as(usize, 2), m_alice.conflictCountOf(shared));
    try t.expectEqual(@as(usize, 2), m_bob.conflictCountOf(bob_shared));

    // Both replicas agree on the single deterministic winner.
    const winner_a = m_alice.parentOf(shared).?;
    const winner_b = m_bob.parentOf(bob_shared).?;
    try t.expect(winner_a == .node);
    try t.expect(winner_b == .node);
    try t.expect(eqId(winner_a.node, parent_a) or eqId(winner_a.node, parent_b));
}

test "concurrent moves that individually are acyclic but jointly cycle: deterministically resolved, stays acyclic, both replicas agree" {
    const gpa = t.allocator;
    var alice: Doc = .empty;
    defer alice.deinit(gpa);
    var bob: Doc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    const k0 = try between(gpa, null, null);
    defer gpa.free(k0);
    const na = try alice.create(gpa, .root, k0);
    const k1 = try between(gpa, k0, null);
    defer gpa.free(k1);
    const nb = try alice.create(gpa, .root, k1);
    try bob.syncFrom(gpa, &alice);

    const bob_na: EventId = .{ .agent = bob.history.findAgent("alice").?, .seq = na.seq };
    const bob_nb: EventId = .{ .agent = bob.history.findAgent("alice").?, .seq = nb.seq };

    // A moves under B (alice) concurrently with B moves under A (bob).
    const ka = try between(gpa, null, null);
    defer gpa.free(ka);
    try alice.move(gpa, na, .{ .node = nb }, ka);
    const kb = try between(gpa, null, null);
    defer gpa.free(kb);
    try bob.move(gpa, bob_nb, .{ .node = bob_na }, kb);

    try bob.syncFrom(gpa, &alice);
    try alice.syncFrom(gpa, &bob);

    var m_alice = try materialize(&alice, gpa);
    defer m_alice.deinit(gpa);
    var m_bob = try materialize(&bob, gpa);
    defer m_bob.deinit(gpa);

    try m_alice.verifyForest(gpa); // acyclic, independently re-derived
    try m_bob.verifyForest(gpa);

    const dump_a = try m_alice.dump(gpa);
    defer gpa.free(dump_a);
    const dump_b = try m_bob.dump(gpa);
    defer gpa.free(dump_b);
    try t.expectEqualStrings(dump_a, dump_b); // both replicas agree

    // Exactly one of the two writes was rejected (both cannot apply).
    try t.expectEqual(@as(usize, 1), m_alice.rejected.items.len);
    try t.expectEqual(@as(usize, 1), m_bob.rejected.items.len);
}

// Pins the corrected FINDING above `materialize`: a reviewer's
// counterexample showed an earlier draft's "the materialized winner is
// always a legitimate conflict-set member" claim is false. Single
// replica, fully sequential (no concurrency at all — this is not about
// MV-register conflicts, it's about cycle-rejection stranding an old
// write): create N; create P; move P under N (accepted); move N under P
// (this WOULD make N its own ancestor via P, so it is rejected). N's
// effective parent falls all the way back to its own `create` value
// (`.root`) — but `conflict_live[N]` was already killed down to just the
// REJECTED move (it causally superseded the `create`, whether or not it
// went on to apply), so the winner sits outside the reported conflict set.
test "a rejected write can leave an earlier, already-superseded write as the effective parent — outside the reported conflict set" {
    const gpa = t.allocator;
    var d: Doc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const kn = try between(gpa, null, null);
    defer gpa.free(kn);
    const n = try d.create(gpa, .root, kn);
    const kp = try between(gpa, kn, null);
    defer gpa.free(kp);
    const p = try d.create(gpa, .root, kp);

    const k1 = try between(gpa, null, null);
    defer gpa.free(k1);
    try d.move(gpa, p, .{ .node = n }, k1); // P under N: fine, no cycle yet

    const k2 = try between(gpa, null, null);
    defer gpa.free(k2);
    try d.move(gpa, n, .{ .node = p }, k2); // N under P: would cycle, rejected

    var m = try materialize(&d, gpa);
    defer m.deinit(gpa);
    try m.verifyForest(gpa); // still acyclic, still fully reachable — convergence unharmed

    try t.expectEqual(@as(usize, 1), m.rejected.items.len);

    // The winner: N's own `create`, all the way back — not the rejected move.
    const winner = m.parentOf(n).?;
    try t.expect(winner == .root);

    // The reported conflict set for N's register, though, contains only
    // the rejected move — `create N` was already killed from it, because
    // the later write causally superseded it regardless of whether that
    // later write was itself accepted. The winner is NOT in this set.
    try t.expectEqual(@as(usize, 1), m.conflictCountOf(n));
    const live = m.conflict_live.get(n).?;
    try t.expectEqual(@as(usize, 1), live.items.len);
    try t.expectEqual(m.rejected.items[0], live.items[0]);
    try t.expect(!eqId(d.history.idOf(live.items[0]), n)); // not `create N` itself
}

test "delete hides a subtree without destroying it; undelete restores it intact" {
    const gpa = t.allocator;
    var d: Doc = .empty;
    defer d.deinit(gpa);
    try d.setAgent(gpa, "solo");

    const k0 = try between(gpa, null, null);
    defer gpa.free(k0);
    const parent = try d.create(gpa, .root, k0);
    const k1 = try between(gpa, null, null);
    defer gpa.free(k1);
    const child = try d.create(gpa, .{ .node = parent }, k1);

    try d.delete(gpa, parent);

    var m1 = try materialize(&d, gpa);
    defer m1.deinit(gpa);
    const roots1 = try m1.childrenOf(gpa, .root);
    defer gpa.free(roots1);
    try t.expectEqual(@as(usize, 0), roots1.len); // hidden from root
    const trash_kids = try m1.childrenOf(gpa, .trash);
    defer gpa.free(trash_kids);
    try t.expectEqual(@as(usize, 1), trash_kids.len);
    try t.expect(eqId(trash_kids[0], parent));
    const under_trashed = try m1.childrenOf(gpa, .{ .node = parent });
    defer gpa.free(under_trashed);
    try t.expectEqual(@as(usize, 1), under_trashed.len);
    try t.expect(eqId(under_trashed[0], child)); // child untouched, still there
    try m1.verifyForest(gpa);

    // Undelete: move the deleted node back under root.
    const k2 = try between(gpa, null, null);
    defer gpa.free(k2);
    try d.move(gpa, parent, .root, k2);

    var m2 = try materialize(&d, gpa);
    defer m2.deinit(gpa);
    const roots2 = try m2.childrenOf(gpa, .root);
    defer gpa.free(roots2);
    try t.expectEqual(@as(usize, 1), roots2.len);
    const restored_kids = try m2.childrenOf(gpa, .{ .node = parent });
    defer gpa.free(restored_kids);
    try t.expectEqual(@as(usize, 1), restored_kids.len);
    try t.expect(eqId(restored_kids[0], child)); // subtree intact
    try m2.verifyForest(gpa);
}

test "random insertions keep keys short (healthy case)" {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    var keys: std.ArrayList([]u8) = .empty;
    defer {
        for (keys.items) |k| gpa.free(k);
        keys.deinit(gpa);
    }
    try keys.append(gpa, try between(gpa, null, null));
    for (0..500) |_| {
        const i = random.uintLessThan(usize, keys.items.len);
        const a: ?[]const u8 = if (i == 0) null else keys.items[i - 1];
        const b: ?[]const u8 = keys.items[i];
        const nk = try between(gpa, a, b);
        try keys.insert(gpa, i, nk);
    }
    var max_len: usize = 0;
    for (keys.items) |k| max_len = @max(max_len, k.len);
    // 501 keys scattered at random gaps: expect small single-digit depth
    // (random insertion does not repeatedly bisect the same shrinking
    // interval), nowhere near the adversarial N/8 bound below.
    try t.expect(max_len <= 6);
}

test "pathological same-gap insertion grows keys linearly, not logarithmically" {
    const gpa = t.allocator;
    // Always insert immediately to the left of the most recent insertion
    // (repeated bisection of one shrinking interval) — see module doc.
    var bound: ?[]u8 = null;
    defer if (bound) |b| gpa.free(b);
    const n = 256;
    var last_len: usize = 0;
    for (0..n) |_| {
        const nk = try between(gpa, null, bound);
        if (bound) |b| gpa.free(b);
        bound = nk;
        last_len = nk.len;
    }
    // ~8 bisections exhaust one byte (256 -> 128 -> ... -> 1), so N
    // insertions cost roughly N/8 bytes. Assert the honest linear
    // ballpark, not a log bound the algorithm cannot deliver.
    const expected = n / 8;
    try t.expect(last_len >= expected / 2 and last_len <= expected * 2 + 4);
}

test "untouched siblings keep relative order across merges" {
    const gpa = t.allocator;
    var alice: Doc = .empty;
    defer alice.deinit(gpa);
    var bob: Doc = .empty;
    defer bob.deinit(gpa);
    try alice.setAgent(gpa, "alice");
    try bob.setAgent(gpa, "bob");

    var prev: ?[]u8 = null;
    var untouched: [3]EventId = undefined;
    for (0..3) |i| {
        const k = try between(gpa, prev, null);
        defer gpa.free(k);
        untouched[i] = try alice.create(gpa, .root, k);
        if (prev) |p| gpa.free(p);
        prev = try gpa.dupe(u8, k);
    }
    if (prev) |p| gpa.free(p);
    try bob.syncFrom(gpa, &alice);

    // Bob concurrently inserts a new sibling somewhere in the middle;
    // alice concurrently deletes an unrelated other node (none of the
    // `untouched` three are touched by anyone).
    var mb = try materialize(&bob, gpa);
    defer mb.deinit(gpa);
    const bob_roots = try mb.childrenOf(gpa, .root);
    defer gpa.free(bob_roots);
    const bk = try between(gpa, mb.doc.keyBytes(mb.states.get(bob_roots[0]).?.key_digits), mb.doc.keyBytes(mb.states.get(bob_roots[1]).?.key_digits));
    defer gpa.free(bk);
    _ = try bob.create(gpa, .root, bk);

    try bob.syncFrom(gpa, &alice);
    try alice.syncFrom(gpa, &bob);

    var m_alice = try materialize(&alice, gpa);
    defer m_alice.deinit(gpa);
    const final_roots = try m_alice.childrenOf(gpa, .root);
    defer gpa.free(final_roots);
    try t.expectEqual(@as(usize, 4), final_roots.len);

    // The three untouched nodes keep their pairwise relative order.
    var positions: [3]usize = undefined;
    for (untouched, 0..) |u, ui| {
        for (final_roots, 0..) |r, ri| {
            if (eqId(r, u)) positions[ui] = ri;
        }
    }
    try t.expect(positions[0] < positions[1]);
    try t.expect(positions[1] < positions[2]);
}

// The property-test campaign: seeded-PRNG random schedules of
// create/move(reorder)/delete/sync across 2-4 replicas, asserting after
// full exchange: convergence (byte-identical canonical dumps) and
// acyclicity + reachability (`verifyForest`, an independent oracle, not
// the construction algorithm's own bookkeeping). Order stability is NOT
// separately checked per schedule here — it's proven by the dedicated
// `"untouched siblings keep relative order across merges"` test above;
// this campaign is merely consistent with it, since keys are never
// mutated once assigned, not additional evidence for it.
//
// 400 seeds x ~30 ops x up to 4 replicas run in this test. During
// development, `Doc.syncFrom`'s first version passed `self`'s (the sync
// destination's) frontier — LOCAL `Lv` indices — directly as the
// `remote_frontier` argument to `from.history.missingFrom`, which indexes
// `from`'s OWN event array with them. This is precisely the mistake the
// module doc warns about ("Lv... NOT stable across replicas"), made in the
// one helper whose whole job is to guard against it. It surfaced
// immediately and everywhere a sync happened more than trivially: the
// hand-written two-replica tests above (concurrent-move-of-one-node,
// concurrent-moves-that-cycle) failed with alice's and bob's materialized
// trees genuinely DISAGREEING (each computed a different effective parent
// for the concurrently-moved node — not a display artifact, a real
// convergence failure), and this property test — bigger graphs, multi-hop
// sync, four replicas — turned the same root cause into an index-out-of-
// bounds panic and a null-unwrap panic on `Lv`s that didn't exist in the
// array they were being read from. All four failures traced to the same
// line. The fix: translate the frontier through portable (agent name,
// seq) into `from`'s local Lv space first, exactly as `core.decodeVersion`
// does for a real version token (see `syncFrom`). Left as the object
// lesson it is — a replica-local-id bug corrupts silently on lucky inputs
// and only some shapes of test surface it as a crash; convergence
// comparison across MULTIPLE replicas is what actually catches it, a
// single-replica self-consistency check would not have.
test "property: random schedules across 2-4 replicas converge, stay acyclic, stay reachable" {
    const gpa = t.allocator;
    const schedules = 400;
    var seed: u64 = 0;
    while (seed < schedules) : (seed += 1) {
        try runSchedule(gpa, seed);
    }
}

fn runSchedule(gpa: Allocator, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const replica_count = 2 + random.uintLessThan(usize, 3); // 2..4

    var docs: [4]Doc = .{ .empty, .empty, .empty, .empty };
    defer for (0..replica_count) |i| docs[i].deinit(gpa);
    const names = [_][]const u8{ "r0", "r1", "r2", "r3" };
    for (0..replica_count) |i| try docs[i].setAgent(gpa, names[i]);

    // Portable node handles: (creator agent name, seq), resolved to each
    // replica's local EventId lazily (only among nodes that replica has
    // actually synced).
    var handles: std.ArrayList(NodeHandle) = .empty;
    defer handles.deinit(gpa);

    const op_count = 30;
    for (0..op_count) |_| {
        const ri = random.uintLessThan(usize, replica_count);
        const d = &docs[ri];
        const choice = random.uintLessThan(u8, 5);
        switch (choice) {
            0, 1 => { // create (weighted higher so there's material to move)
                const parent = (try pickTarget(gpa, d, random, handles.items, true)).?;
                var m = try materialize(d, gpa);
                defer m.deinit(gpa);
                const kids = try m.childrenOf(gpa, parent);
                defer gpa.free(kids);
                const digits = try randomGapKey(gpa, d, &m, kids, random);
                defer gpa.free(digits);
                const id = try d.create(gpa, parent, digits);
                try handles.append(gpa, .{ .name = d.history.agentName(id.agent), .seq = id.seq });
            },
            2 => { // move / reorder
                const target = try pickTarget(gpa, d, random, handles.items, false) orelse continue;
                if (target != .node) continue;
                const parent = (try pickTarget(gpa, d, random, handles.items, true)).?;
                var m = try materialize(d, gpa);
                defer m.deinit(gpa);
                const kids = try m.childrenOf(gpa, parent);
                defer gpa.free(kids);
                const digits = try randomGapKey(gpa, d, &m, kids, random);
                defer gpa.free(digits);
                try d.move(gpa, target.node, parent, digits);
            },
            3 => { // delete
                const target = try pickTarget(gpa, d, random, handles.items, false) orelse continue;
                if (target != .node) continue;
                try d.delete(gpa, target.node);
            },
            4 => { // sync a random pair
                if (replica_count < 2) continue;
                var i = random.uintLessThan(usize, replica_count);
                var j = random.uintLessThan(usize, replica_count);
                if (i == j) j = (j + 1) % replica_count;
                try docs[i].syncFrom(gpa, &docs[j]);
                i = random.uintLessThan(usize, replica_count);
                j = random.uintLessThan(usize, replica_count);
                if (i == j) j = (j + 1) % replica_count;
                try docs[i].syncFrom(gpa, &docs[j]);
            },
            else => unreachable,
        }
    }

    // Full exchange.
    for (0..2) |_| {
        for (0..replica_count) |i| {
            for (0..replica_count) |j| {
                if (i != j) try docs[i].syncFrom(gpa, &docs[j]);
            }
        }
    }

    var dumps: [4][]u8 = undefined;
    defer for (0..replica_count) |i| gpa.free(dumps[i]);
    for (0..replica_count) |i| {
        var m = try materialize(&docs[i], gpa);
        defer m.deinit(gpa);
        try m.verifyForest(gpa); // acyclic + fully reachable, every replica
        dumps[i] = try m.dump(gpa);
    }
    for (1..replica_count) |i| {
        std.testing.expectEqualStrings(dumps[0], dumps[i]) catch |err| {
            std.debug.print("seed={d} replica_count={d} mismatch at replica {d}\n", .{ seed, replica_count, i });
            return err;
        };
    }
}

/// Pick a `.root`/`.trash`/known-node target for replica `d`, from the
/// portable `handles` list, filtered to ones `d` has actually synced.
/// `allow_special` includes `.root`/`.trash` in the pool; when false and no
/// node is known yet, returns `null`.
fn pickTarget(
    gpa: Allocator,
    d: *const Doc,
    random: std.Random,
    handles: []const NodeHandle,
    allow_special: bool,
) !?NodeRef {
    var known: std.ArrayList(EventId) = .empty;
    defer known.deinit(gpa);
    for (handles) |h| {
        if (d.history.findAgent(h.name)) |aid| {
            const id: EventId = .{ .agent = aid, .seq = h.seq };
            if (d.history.isKnown(id)) try known.append(gpa, id);
        }
    }
    if (allow_special) {
        const pool = known.items.len + 2; // + root + trash
        const pick = random.uintLessThan(usize, pool);
        if (pick == 0) return .root;
        if (pick == 1) return .trash;
        return .{ .node = known.items[pick - 2] };
    }
    if (known.items.len == 0) return null;
    return .{ .node = known.items[random.uintLessThan(usize, known.items.len)] };
}

fn randomGapKey(gpa: Allocator, d: *const Doc, m: *const Materialized, kids: []const EventId, random: std.Random) ![]u8 {
    if (kids.len == 0) return between(gpa, null, null);
    const gap = random.uintLessThan(usize, kids.len + 1);
    var a: ?[]const u8 = if (gap == 0) null else d.keyBytes(m.states.get(kids[gap - 1]).?.key_digits);
    var b: ?[]const u8 = if (gap == kids.len) null else d.keyBytes(m.states.get(kids[gap]).?.key_digits);
    // Two siblings can tie on raw digits, distinguished only by the
    // key_writer tiebreak (module doc: concurrent identical-bound inserts
    // — e.g. two replicas both appending — commonly compute the exact
    // same midpoint). `between()`'s precondition requires strictly-ordered
    // bounds, so widen past the whole tied run to the nearest genuinely
    // distinct neighbor (or `null`, if the tie runs to an end).
    if (a != null and b != null and std.mem.eql(u8, a.?, b.?)) {
        var lo = gap;
        while (lo > 0 and std.mem.eql(u8, d.keyBytes(m.states.get(kids[lo - 1]).?.key_digits), a.?)) lo -= 1;
        var hi = gap;
        while (hi < kids.len and std.mem.eql(u8, d.keyBytes(m.states.get(kids[hi]).?.key_digits), b.?)) hi += 1;
        a = if (lo == 0) null else d.keyBytes(m.states.get(kids[lo - 1]).?.key_digits);
        b = if (hi == kids.len) null else d.keyBytes(m.states.get(kids[hi]).?.key_digits);
    }
    return between(gpa, a, b);
}

test {
    std.testing.refAllDecls(@This());
}
