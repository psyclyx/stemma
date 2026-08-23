//! ObjectDoc — a collaborative JSON document over the event graph: the
//! Google-Docs-shaped object model. A tree of maps, lists, inline scalars,
//! and full text objects (each text node is a real sequence CRDT with the
//! same FugueMax semantics as `TextDoc`), all sharing one causal history,
//! one wire format, one sync protocol.
//!
//! Consumer contract:
//! - Local edits: `mapSet`/`mapDelete`/`listInsert`/`listDelete`/
//!   `textInsert`/`textDelete`, addressed by `ObjId` (`null` = root map).
//!   Creating values (`.map`/`.list`/`.text`) returns the new object's id.
//! - Reads: `root()` → `ValueRef` navigation. Map reads give a
//!   deterministic winner plus the honest multi-value conflict set
//!   (concurrent sets both survive — the only truthful semantics in a
//!   clockless system; resolve conflicts by policy or by writing again).
//! - `merge` returns a `[]Change` stream: map key touches, list index
//!   edits, and byte-space text `Edit`s per text object (shift your
//!   per-object anchors exactly as with `TextDoc`).
//! - `version`/`eventsSince`/`compareVersions`/`serialize`/`open`: same
//!   opaque tokens and sync model as `TextDoc`.
//!
//! Not yet (ledgered): compaction for ObjectDoc, identity anchors inside
//! ObjectDoc text objects, Peritext-style rich-text marks (representable
//! today as mark objects holding anchor values), incremental persistence.
//! Unification of the TextDoc/ObjectDoc doc-core scaffolding is a known
//! cleanup.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const jw = @import("objects_state.zig");
const core = @import("core.zig");
const wire = @import("wire.zig");
const seq_walker = @import("SeqWalker.zig");
const rope_mod = @import("../rope.zig");
const geometry = @import("../geometry.zig");

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
pub const ObjId = jw.ObjId;
const Lv = causal.Lv;
const Graph = jw.Graph;
const Walker = jw.Walker;
const ObjectOp = jw.ObjectOp;
const ValPayload = jw.ValPayload;
const Str = jw.Str;
const Effect = jw.Effect;
const SeqWalker = jw.SeqWalker;
/// Where a structural node's parent register can point (F3, delta 6): the
/// two permanent roots, or another structural node by portable identity.
/// See `structCreate`/`structMove`/`structParent`.
pub const StructRef = jw.StructRef;
/// Byte-string fractional order-key midpoint — see `structCreate`'s doc
/// comment for the sibling-ordering contract (`structChildren`'s sort
/// key) and `structure_sketch.zig`'s module doc for the full growth-bound
/// discussion this port carries unchanged.
pub const orderKeyBetween = jw.between;
const Rope = rope_mod.Rope;
const Range = geometry.Range;
const Edit = geometry.Edit;
const putUv = wire.putUv;
const getUv = wire.getUv;
const getBytes = wire.getBytes;
const versionSingleEntry = core.versionSingleEntry;

const object_magic_v1 = "stj\x01";
/// v2: emitted whenever `base_version.len > 0` — TWO producers, both
/// meaning "this doc carries a compacted base": `compact` (delta 2) is
/// the original one; `openFromContent`'s bulk load (W7-1) is the second,
/// setting `base_version`/`text_bases` directly without ever calling
/// `compact` itself (see `openFromContent`'s doc comment) — the wire
/// doesn't distinguish "compacted via history" from "born compacted",
/// only "has a base or not", so both route through the same v2 encoding
/// unmodified. A doc that has neither compacted nor been bulk-loaded
/// always emits `object_magic_v1` bytes, byte-identical to pre-delta-2
/// output — old decoders (and every hand-written wire-byte test) keep
/// working unchanged. See the "Wire format" section below for the exact
/// v1/v2 layout diff.
const object_magic_v2 = "stj\x02";
/// v3: emitted instead of v2 whenever the sender has any active hole
/// (`text_holes`, partial checkout — `openPartial`'s section doc comment
/// above `openPartial` has the full wire-contract rationale). Same layout
/// as v2 except the base section's `text_base_count` is always 0 (its
/// entries omitted entirely, not merely empty-content) — a v3 base
/// section carries only `base_version`, enough for a same-base peer to
/// merge, NEVER enough to bootstrap a fresh replica. A doc that never
/// partial-checked-out never emits v3.
const object_magic_v3 = "stj\x03";
const version_magic = core.version_magic;
const no_node: u32 = std.math.maxInt(u32);
const root_key = Walker.root_key;
/// Wire cap on a structural order key's byte length (`Decoder`'s
/// `getBytes(&cur, max_order_key_len)` for `struct_create`/`struct_move`)
/// — asserted at the ORIGINATING end too (`structCreate`/`structMove`),
/// so a key can never be written that this same decoder couldn't read
/// back. Not expected to bind in practice: `orderKeyBetween`'s growth is
/// ~N/8 bytes even under the adversarial same-gap pattern (see its doc
/// comment) — a key anywhere near this cap is a sign the periodic
/// sibling-key rebalancing named as owed (F3 caveat 2, `compact`'s doc
/// comment on structural ops) is overdue, not a case this wire format
/// tries to accommodate unbounded.
const max_order_key_len: usize = 4096;

const ObjectDoc = @This();
/// Input values for local edits.
pub const Value = union(enum) {
    null_,
    bool_: bool,
    int: i64,
    float: f64,
    str: []const u8,
    map,
    list,
    text,
};

pub const Kind = enum { null_, bool_, int, float, str, map, list, text };

pub const Change = union(enum) {
    /// A key's value set changed (added, removed, or overwritten) —
    /// re-read via `mapGet`/`mapConflicts`. `key` borrows the doc's
    /// string arena (valid for the doc's lifetime).
    map: struct { obj: ?ObjId, key: []const u8 },
    list_ins: struct { obj: ObjId, index: usize },
    list_del: struct { obj: ObjId, index: usize },
    /// Byte-space edit within a text object; shift that object's anchors.
    text: struct { obj: ObjId, edit: Edit },
    /// A structural node's effective parent, or its cycle-break status,
    /// may have changed — re-read via `structParent`/`structCycleBroken`/
    /// `structConflictCount` (F3, delta 6). See `structParent`'s doc
    /// comment for the landmine: this is a DIFFERENT conflict-resolution
    /// rule than the plain `map` case above.
    structure: struct { node: ObjId },
};

history: Graph = .empty,
agent: ?AgentId = null,
/// Append-only arena for keys and string values (Str refs point here).
strings: std.ArrayList(u8) = .empty,
/// Materialized tree. Node 0 (once created) is the root map.
nodes: std.ArrayList(Node) = .empty,
/// Creation-event lv → node index (no_node until materialized). ONLY
/// valid against `self.history`'s CURRENT `Lv` space — a transient,
/// replay-time index (`nodeOfObjLv`), rebuilt at the new size on every
/// `compact`. Never consult it with an `Lv` obtained before a `compact`
/// call; that's exactly what `obj_index` is for.
node_of: std.ArrayList(u32) = .empty,
/// Creation `EventId` → node index — the STABLE counterpart to `node_of`,
/// keyed by portable identity rather than `Lv`. Populated once per
/// creation (`makeValueNode`) and NEVER touched by `compact` (an
/// `EventId` doesn't change when its creating event's `Lv` is renumbered
/// or folded into a base — see `compact`'s doc comment). This is what
/// makes an `ObjId` obtained before compaction still resolve afterward.
obj_index: std.AutoHashMapUnmanaged(EventId, u32) = .empty,

/// Compaction (delta 2, `stemma-unification.md` §3 step 4). Whole-doc,
/// one linearization point — see `compact`'s doc comment for the exact
/// contract: only `text_ins`/`text_del` ever fold away; map registers and
/// list structure never do. `base_version.len == 0` = never compacted
/// (every doc's initial state; the overwhelming common case, and the
/// ONLY state pre-delta-2 code ever saw).
base_version: []u8 = &.{},
/// The single stable head `base_version` was compacted at, once compacted.
base_head: ?EventId = null,
/// Per-text-object compacted bases — see `jw.TextBase`. Keyed by the
/// text object's portable creation identity (`jw.TextBaseMap`'s doc
/// comment).
text_bases: jw.TextBaseMap = .empty,
/// Per-text-object unrealized spans of `text_bases` (`openPartial` —
/// see the "Partial checkout" section below). Keyed exactly like
/// `text_bases` (the object's portable creation identity), NOT doc-wide:
/// each text object's holes are its own — the per-object analog of
/// `TextDoc.holes`. A hashmap MISS (the overwhelming common case: no
/// object has ever gone through `openPartial`) costs one lookup, same
/// zero-cost-when-untouched shape `text_bases` itself already has.
/// Absent entirely for an object with no active holes (an object realized
/// down to zero holes is REMOVED from this map, not left with an empty
/// list — `realizeBase` and `baseRealized`/`unrealizedBase` rely on this).
/// Each present entry's `ArrayList(BaseHole)` is ascending by
/// `cur_offset`, same invariant as `TextDoc.holes`.
text_holes: std.AutoHashMapUnmanaged(EventId, std.ArrayList(BaseHole)) = .empty,

/// Structural parent-register placements (F3, delta 6): a node's
/// PORTABLE creation identity → its currently RESOLVED (effective)
/// placement. Populated by `structCreate`/`structMove` (local, always
/// single-writer — see `wouldCycleLocal`) and by `merge`'s
/// `.struct_parent` effect handler (remote, via `Walker`'s GLOBAL
/// Lamport-canonical resolution — see `objects_state.Walker.resolveStructs`
/// and, at this API boundary, `structParent`'s doc comment for how its
/// winner rule differs from `mapGet`'s). `EventId`-keyed like
/// `obj_index`/`text_bases` — stable across `compact` (structural ops
/// currently refuse compaction outright, see `compact`'s doc comment, so
/// this is defense-in-depth rather than load-bearing today).
struct_parents: std.AutoHashMapUnmanaged(EventId, StructPlacement) = .empty,

pub const empty: ObjectDoc = .{};

/// Walker-effect-space-adjacent, but keyed by portable `EventId` (this is
/// ObjectDoc's OWN materialized copy, not Walker's transient Lv-keyed
/// state).
const StructPlacementRef = union(enum) { root, trash, node: EventId };
const StructPlacement = struct {
    parent: StructPlacementRef,
    order_key: Str,
    key_writer: EventId,
    conflict_count: u32,
    cycle_broken: bool,
};

fn toPlacementRef(r: StructRef) StructPlacementRef {
    return switch (r) {
        .root => .root,
        .trash => .trash,
        .node => |id| .{ .node = id },
    };
}

fn structRefEql(a: StructPlacementRef, b: StructPlacementRef) bool {
    return switch (a) {
        .root => b == .root,
        .trash => b == .trash,
        .node => |na| switch (b) {
            .node => |nb| std.meta.eql(na, nb),
            else => false,
        },
    };
}

/// `Unrealized` (delta: partial checkout, `openPartial`) is additive to
/// this set — every stemma-internal call site that exhaustively `switch`es
/// a `MergeError`-shaped catch was audited when it was added (none needed
/// a new arm: they all either `try`-propagate or already carry a wildcard
/// `else => |err| return err`, see `ObjectDoc.zig`'s partial-checkout
/// section doc comment for the full list). Weft's own call sites are the
/// orchestrator's problem at the next stemma pin bump.
pub const MergeError = Allocator.Error || error{ Corrupt, MissingDependency, Unrealized };
pub const CompactError = MergeError || error{NotCompactable};
pub const VersionOrder = causal.VersionOrder;

// Identity anchors (delta 3, stemma-unification.md §3 step 3): same
// portable shape as `TextDoc.EventAnchor` (agent name + seq + side), same
// shared machinery (`SeqWalker.zig`'s `anchorAt`/`resolveAnchors`), one
// per-text-object `SeqWalker` instantiation instead of TextDoc's one
// whole-document instantiation. See `objectAnchorAt`/`resolveObjectAnchors`
// below.
pub const AnchorSide = seq_walker.AnchorSide;
pub const EventAnchor = seq_walker.EventAnchor;
pub const AnchorError = seq_walker.AnchorError;

const ScalarNode = union(enum) { null_, bool_: bool, int: i64, float: f64, str: Str };
// `TreeVal.set_id` and every `Node` variant's `obj` are the PORTABLE
// identity (`EventId` = agent + seq) of the event that created them, not
// the transient `Lv` of the moment — see `compact`'s doc comment: once
// map/list/text creation events can themselves be folded into a
// compacted base, a raw `Lv` stops being a valid handle (it may no
// longer name any node in `self.history` at all), while `EventId` is
// stable forever (compaction renumbers `Lv`s; it never changes an
// agent's registration order or an event's `seq`). `self.obj_index`
// (`EventId` → node index) is the read-direction counterpart, populated
// once per creation and never touched by `compact`.
const TreeVal = struct { set_id: EventId, node: u32 };
const TreeSlot = struct { key: Str, values: std.ArrayList(TreeVal) = .empty };
const Node = union(enum) {
    scalar: ScalarNode,
    map: struct { obj: ?EventId, slots: std.ArrayList(TreeSlot) = .empty },
    list: struct { obj: EventId, elems: std.ArrayList(u32) = .empty },
    text: struct { obj: EventId, rope: Rope = .empty },
};

pub fn deinit(self: *ObjectDoc, gpa: Allocator) void {
    for (self.nodes.items) |*n| switch (n.*) {
        .map => |*m| {
            for (m.slots.items) |*s| s.values.deinit(gpa);
            m.slots.deinit(gpa);
        },
        .list => |*l| l.elems.deinit(gpa),
        .text => |*t| t.rope.deinit(gpa),
        .scalar => {},
    };
    self.nodes.deinit(gpa);
    self.node_of.deinit(gpa);
    self.obj_index.deinit(gpa);
    self.strings.deinit(gpa);
    self.history.deinit(gpa);
    gpa.free(self.base_version);
    var bit = self.text_bases.valueIterator();
    while (bit.next()) |b| gpa.free(b.bytes);
    self.text_bases.deinit(gpa);
    var hit = self.text_holes.valueIterator();
    while (hit.next()) |h| h.deinit(gpa);
    self.text_holes.deinit(gpa);
    self.struct_parents.deinit(gpa);
    self.* = .{};
}

pub fn setAgent(self: *ObjectDoc, gpa: Allocator, name: []const u8) Allocator.Error!void {
    self.agent = try self.history.registerAgent(gpa, name);
}

/// Raw event count since the last compaction (or since genesis,
/// uncompacted) — replay cost scales with this. Delegates to `history`
/// so callers never need to reach through `ObjectDoc`'s internal `Graph`
/// field.
pub fn eventCount(self: *const ObjectDoc) usize {
    return self.history.eventCount();
}

// ObjId values are DOC-LOCAL handles (they embed replica-local agent
// numbering) — never transport one to another replica. To reference an
// object across peers, either navigate document structure (paths) or
// exchange an exported token:

/// Portable object reference token ("sto" 0x01, name, seq). Caller owns.
pub fn exportId(self: *const ObjectDoc, gpa: Allocator, obj: ObjId) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "sto\x01");
    const name = self.history.agentName(obj.agent);
    try putUv(gpa, &out, name.len);
    try out.appendSlice(gpa, name);
    try putUv(gpa, &out, obj.seq);
    return out.toOwnedSlice(gpa);
}

/// Resolve a token from `exportId` (possibly minted on another replica)
/// to this replica's handle. `error.MissingDependency` if we have not
/// seen the creating event.
pub fn importId(self: *const ObjectDoc, token: []const u8) (error{ Corrupt, MissingDependency })!ObjId {
    var cur = token;
    if (!std.mem.startsWith(u8, cur, "sto\x01")) return error.Corrupt;
    cur = cur[4..];
    const name = try getBytes(&cur, 4096);
    const seq = try getUv(&cur);
    const agent = self.history.findAgent(name) orelse return error.MissingDependency;
    const id: EventId = .{ .agent = agent, .seq = seq };
    if (self.history.lvOf(id) == null) return error.MissingDependency;
    return id;
}

fn str(self: *const ObjectDoc, s: Str) []const u8 {
    return self.strings.items[s.start..][0..s.len];
}

fn intern(self: *ObjectDoc, gpa: Allocator, bytes: []const u8) Allocator.Error!Str {
    const start: u32 = @intCast(self.strings.items.len);
    try self.strings.appendSlice(gpa, bytes);
    return .{ .start = start, .len = @intCast(bytes.len) };
}

fn ensureRoot(self: *ObjectDoc, gpa: Allocator) Allocator.Error!void {
    if (self.nodes.items.len == 0) {
        try self.nodes.append(gpa, .{ .map = .{ .obj = null } });
    }
}

fn ensureNodeMap(self: *ObjectDoc, gpa: Allocator) Allocator.Error!void {
    const n = self.history.eventCount();
    if (self.node_of.items.len < n) {
        try self.node_of.appendNTimes(gpa, no_node, n - self.node_of.items.len);
    }
}

/// `obj`'s node index, valid ONLY against `self.history`'s CURRENT `Lv`
/// space — used strictly transiently, during local edits or one replay
/// pass, always with an `Lv` obtained THIS SAME call (never one cached
/// across a `compact`).
fn nodeOfObjLv(self: *const ObjectDoc, obj: Lv) u32 {
    return if (obj == root_key) 0 else self.node_of.items[obj];
}

/// `obj`'s node index by PORTABLE identity — safe against an `ObjId`
/// obtained at any point in the past, compacted away or not (see
/// `obj_index`'s doc comment).
fn resolveObjNode(self: *const ObjectDoc, obj: ?ObjId) u32 {
    const id = obj orelse return 0;
    return self.obj_index.get(id).?;
}

/// Create the tree node for a freshly applied value payload. `creation_lv`
/// is this replica's CURRENT `Lv` for the creating event — used only to
/// look up its portable `EventId` (stored on the node, and as
/// `obj_index`'s key) and to size-index the transient `node_of`; never
/// itself retained past this call.
fn makeValueNode(self: *ObjectDoc, gpa: Allocator, val: ValPayload, creation_lv: Lv) Allocator.Error!u32 {
    const idx: u32 = @intCast(self.nodes.items.len);
    const id = self.history.idOf(creation_lv);
    const node: Node = switch (val) {
        .null_ => .{ .scalar = .null_ },
        .bool_ => |b| .{ .scalar = .{ .bool_ = b } },
        .int => |v| .{ .scalar = .{ .int = v } },
        .float => |v| .{ .scalar = .{ .float = v } },
        .str => |s| .{ .scalar = .{ .str = s } },
        .new_map => .{ .map = .{ .obj = id } },
        .new_list => .{ .list = .{ .obj = id } },
        .new_text => .{ .text = .{ .obj = id } },
    };
    try self.nodes.append(gpa, node);
    errdefer _ = self.nodes.pop();
    switch (val) {
        .new_map, .new_list, .new_text => {
            self.node_of.items[creation_lv] = idx;
            try self.obj_index.put(gpa, id, idx);
        },
        else => {},
    }
    return idx;
}

fn treeSlot(self: *ObjectDoc, gpa: Allocator, map_node: u32, key: Str) Allocator.Error!*TreeSlot {
    const m = &self.nodes.items[map_node].map;
    for (m.slots.items) |*s| {
        if (std.mem.eql(u8, self.str(s.key), self.str(key))) return s;
    }
    try m.slots.append(gpa, .{ .key = key });
    return &m.slots.items[m.slots.items.len - 1];
}

// ── Local editing ───────────────────────────────────────────────────

fn payloadFrom(self: *ObjectDoc, gpa: Allocator, v: Value) Allocator.Error!ValPayload {
    return switch (v) {
        .null_ => .null_,
        .bool_ => |b| .{ .bool_ = b },
        .int => |x| .{ .int = x },
        .float => |x| .{ .float = x },
        .str => |s| .{ .str = try self.intern(gpa, s) },
        .map => .new_map,
        .list => .new_list,
        .text => .new_text,
    };
}

/// Set `key` in map `obj` (`null` = root). Overwrites every value the
/// local replica currently sees. Returns the created object's id when
/// `val` is `.map`/`.list`/`.text`.
pub fn mapSet(self: *ObjectDoc, gpa: Allocator, obj: ?ObjId, key: []const u8, val: Value) Allocator.Error!?ObjId {
    const agent = self.agent.?;
    try self.ensureRoot(gpa);
    const map_node = self.resolveObjNode(obj);
    assert(self.nodes.items[map_node] == .map);
    const key_str = try self.intern(gpa, key);
    const payload = try self.payloadFrom(gpa, val);
    const lv = try self.history.addLocal(gpa, agent, .{ .map_set = .{ .obj = obj, .key = key_str, .val = payload } });
    try self.ensureNodeMap(gpa);
    const value_node = try self.makeValueNode(gpa, payload, lv);
    const slot = try self.treeSlot(gpa, map_node, key_str);
    slot.values.clearRetainingCapacity(); // local set overwrites all seen
    try slot.values.append(gpa, .{ .set_id = self.history.idOf(lv), .node = value_node });
    return switch (payload) {
        .new_map, .new_list, .new_text => self.history.idOf(lv),
        else => null,
    };
}

pub fn mapDelete(self: *ObjectDoc, gpa: Allocator, obj: ?ObjId, key: []const u8) Allocator.Error!void {
    const agent = self.agent.?;
    try self.ensureRoot(gpa);
    const map_node = self.resolveObjNode(obj);
    assert(self.nodes.items[map_node] == .map);
    const key_str = try self.intern(gpa, key);
    _ = try self.history.addLocal(gpa, agent, .{ .map_del = .{ .obj = obj, .key = key_str } });
    try self.ensureNodeMap(gpa);
    const slot = try self.treeSlot(gpa, map_node, key_str);
    slot.values.clearRetainingCapacity();
}

/// Insert into list `obj` at `index`. Returns the created object's id
/// for `.map`/`.list`/`.text` values.
pub fn listInsert(self: *ObjectDoc, gpa: Allocator, obj: ObjId, index: usize, val: Value) Allocator.Error!?ObjId {
    const agent = self.agent.?;
    try self.ensureRoot(gpa);
    const list_node = self.resolveObjNode(obj);
    const l = &self.nodes.items[list_node].list;
    assert(index <= l.elems.items.len);
    const payload = try self.payloadFrom(gpa, val);
    const lv = try self.history.addLocal(gpa, agent, .{ .list_ins = .{ .obj = obj, .pos = index, .val = payload } });
    try self.ensureNodeMap(gpa);
    const value_node = try self.makeValueNode(gpa, payload, lv);
    try self.nodes.items[list_node].list.elems.insert(gpa, index, value_node);
    return switch (payload) {
        .new_map, .new_list, .new_text => self.history.idOf(lv),
        else => null,
    };
}

pub fn listDelete(self: *ObjectDoc, gpa: Allocator, obj: ObjId, index: usize) Allocator.Error!void {
    const agent = self.agent.?;
    const list_node = self.resolveObjNode(obj);
    const l = &self.nodes.items[list_node].list;
    assert(index < l.elems.items.len);
    _ = try self.history.addLocal(gpa, agent, .{ .list_del = .{ .obj = obj, .pos = index } });
    _ = self.nodes.items[list_node].list.elems.orderedRemove(index);
}

/// Insert UTF-8 `content` into text object `obj` at `byte_offset`.
/// Same contract as `Rope.insert`. Precondition: the offset is not
/// interior to an unrealized base span of `obj` (deterministic panic, all
/// build modes — `realizeBase` first). Cost when `obj` has no active
/// holes (the overwhelming common case): one `text_holes` hashmap miss.
pub fn textInsert(self: *ObjectDoc, gpa: Allocator, obj: ObjId, byte_offset: usize, content: []const u8) Allocator.Error!Edit {
    const agent = self.agent.?;
    const text_node = self.resolveObjNode(obj);
    const t = &self.nodes.items[text_node].text;
    const edit: Edit = .{ .offset = byte_offset, .removed = 0, .inserted = content.len };
    if (content.len == 0) return edit;
    self.assertOutsideHoles(obj, byte_offset, byte_offset);
    const scalar_pos = t.rope.offsetToScalar(byte_offset) + self.holeScalarsThrough(obj, byte_offset);
    var i: u64 = 0;
    var it = (std.unicode.Utf8View.init(content) catch unreachable).iterator();
    while (it.nextCodepoint()) |ch| : (i += 1) {
        _ = try self.history.addLocal(gpa, agent, .{ .text_ins = .{ .obj = obj, .pos = scalar_pos + i, .ch = ch } });
    }
    _ = try self.nodes.items[text_node].text.rope.insert(gpa, byte_offset, content);
    self.shiftHoles(obj, byte_offset, content.len, 0);
    return edit;
}

/// Precondition: the range does not intersect an unrealized base span of
/// `obj` (deterministic panic, all build modes — `realizeBase` first).
pub fn textDelete(self: *ObjectDoc, gpa: Allocator, obj: ObjId, range: Range) Allocator.Error!Edit {
    const agent = self.agent.?;
    const text_node = self.resolveObjNode(obj);
    const t = &self.nodes.items[text_node].text;
    const edit: Edit = .{ .offset = range.start, .removed = range.len(), .inserted = 0 };
    if (range.isEmpty()) return edit;
    self.assertOutsideHoles(obj, range.start, range.end);
    const scalar_start = t.rope.offsetToScalar(range.start) + self.holeScalarsThrough(obj, range.start);
    const scalar_count = t.rope.offsetToScalar(range.end) - t.rope.offsetToScalar(range.start);
    for (0..scalar_count) |_| {
        _ = try self.history.addLocal(gpa, agent, .{ .text_del = .{ .obj = obj, .pos = scalar_start } });
    }
    _ = try self.nodes.items[text_node].text.rope.delete(gpa, range);
    self.shiftHoles(obj, range.start, 0, range.len());
    return edit;
}

/// Panic if `[start, end]` touches the interior of an unrealized base
/// span of `obj` (or covers one). Single choke point, all build modes —
/// mirrors `Rope`'s own hole-content panic and `TextDoc.assertOutsideHoles`,
/// scoped per object.
fn assertOutsideHoles(self: *const ObjectDoc, obj: ObjId, start: usize, end: usize) void {
    const holes = self.text_holes.get(obj) orelse return;
    for (holes.items) |h| {
        if (end > h.cur_offset and start < h.cur_offset + h.bytes) {
            @panic("stemma.ObjectDoc: edit touches an unrealized base span — realizeBase() first");
        }
    }
}

/// Scalars of `obj`'s unrealized base spans at or before `byte_offset`
/// (spans ending exactly at the offset count; spans starting there do
/// not) — see `TextDoc.holeScalarsThrough`.
fn holeScalarsThrough(self: *const ObjectDoc, obj: ObjId, byte_offset: usize) u64 {
    const holes = self.text_holes.get(obj) orelse return 0;
    var acc: u64 = 0;
    for (holes.items) |h| {
        if (h.cur_offset + h.bytes <= byte_offset) acc += h.scalars;
    }
    return acc;
}

/// Shift `obj`'s hole positions through a byte edit at `at`. Holes never
/// intersect edits (checked by `assertOutsideHoles`, or, for a remote
/// merge effect, `checkHoleConflicts`), so a whole-span shift is exact —
/// see `TextDoc.shiftHoles`.
fn shiftHoles(self: *ObjectDoc, obj: ObjId, at: usize, inserted: usize, removed: usize) void {
    const holes = self.text_holes.getPtr(obj) orelse return;
    for (holes.items) |*h| {
        if (h.cur_offset >= at) h.cur_offset = h.cur_offset + inserted - removed;
    }
}

// ── Hole-aware scalar ⇄ byte mapping (merge application) ────────────
// Global scalar positions count unrealized base scalars (the FugueMax
// sequence's own space, seeded by `TextBase.scalars` — the whole count,
// hole or not); `obj`'s rope's OWN scalar metric counts REALIZED content
// only (a hole leaf's summary carries zero scalars — see `rope.zig`'s
// `newLeafHole`). `checkHoleConflicts` already proved a merge effect's
// position never lands INTERIOR to a hole, so mapping only has to
// subtract hole scalars and pin boundary cases to the hole's edge —
// direct per-object ports of `TextDoc.insByteOffset`/`delByteRange`.

/// Byte offset for inserting at global scalar position `pos` within
/// `obj`'s current rope. Safe to call unconditionally: degenerates to
/// plain `rope.scalarToOffset(pos)` when `obj` has no active holes (used
/// that way by `resolveObjectAnchors` below, not just `merge`'s
/// holes-guarded insert path).
fn insByteOffset(self: *const ObjectDoc, obj: ObjId, pos: u64) usize {
    const rope = &self.nodes.items[self.obj_index.get(obj).?].text.rope;
    const holes = self.text_holes.get(obj) orelse return rope.scalarToOffset(@intCast(pos));
    var acc: u64 = 0; // hole scalars strictly before pos
    for (holes.items) |h| {
        const start = @as(u64, rope.offsetToScalar(h.cur_offset)) + acc;
        if (pos <= start) {
            if (pos == start) return h.cur_offset; // insert before the hole
            break;
        }
        acc += h.scalars; // non-interior ⇒ pos ≥ start + scalars
    }
    return rope.scalarToOffset(@intCast(pos - acc));
}

/// Byte range of the (realized) scalar at global position `pos` within
/// `obj`'s current rope, clipped so it never swallows a neighboring
/// hole's bytes. Only ever consulted when `obj` has active holes.
fn delByteRange(self: *const ObjectDoc, obj: ObjId, pos: u64) Range {
    const rope = &self.nodes.items[self.obj_index.get(obj).?].text.rope;
    const holes = self.text_holes.get(obj).?;
    var acc: u64 = 0;
    var clamp: ?usize = null; // first hole after the target
    for (holes.items) |h| {
        const start = @as(u64, rope.offsetToScalar(h.cur_offset)) + acc;
        if (pos < start) {
            clamp = h.cur_offset;
            break;
        }
        acc += h.scalars;
    }
    const rs: usize = @intCast(pos - acc);
    const start = rope.scalarToOffset(rs);
    var end = rope.scalarToOffset(rs + 1);
    if (clamp) |c| end = @min(end, c);
    return .{ .start = start, .end = end };
}

// ── Structural editing (F3, delta 6 — the move op) ────────────────────
// `stemma-unification.md` §3 step 5, ported from `structure_sketch.zig`.
// A structural node's identity is its `struct_create` event, exactly like
// every other object kind; it ALSO behaves as an ordinary map object (see
// `objects_state.Walker.resolveObj`'s `.struct_create` case), so
// `mapSet`/`mapGet`/etc against a structural node's `ObjId` work
// unmodified — this facility is purely about WHERE the node sits in a
// separate, reparentable tree (parent + fractional order key), never
// about its own properties. Lists remain the right shape for leaf
// sequences that never reparent (F3's rationale) — do not use these ops
// to reorder ordinary list elements.
//
// Local edits below are always single-writer (no concurrent sibling can
// exist yet, since a local write's causal parents are the current
// frontier — see `EventGraph.addLocal`), so `wouldCycleLocal` is the only
// check needed here; a REMOTE batch's concurrent/conflicting writes are
// resolved by `objects_state.Walker`'s global Lamport-canonical replay
// during `merge` (see `structParent`'s doc comment for how that rule
// differs from `mapGet`'s).

/// Would giving `node` the parent `parent` make `node` its own ancestor,
/// given the CURRENTLY materialized structural state? Mirrors
/// `objects_state.Walker.wouldCycleStruct`/`structure_sketch.zig:
/// wouldCycle`, operating on `self.struct_parents` (portable
/// `EventId`-keyed) instead of a Walker's transient Lv-keyed state — the
/// two must stay behaviorally identical, or a local edit's immediately
/// materialized view could diverge from what a later `merge`'s full
/// replay reconstructs from the same history.
fn wouldCycleLocal(self: *const ObjectDoc, node: ObjId, parent: StructRef) bool {
    if (parent != .node) return false;
    var cur = parent.node;
    if (std.meta.eql(cur, node)) return true;
    var guard: usize = 0;
    while (true) {
        guard += 1;
        assert(guard <= self.struct_parents.count() + 2);
        const st = self.struct_parents.get(cur) orelse return false;
        switch (st.parent) {
            .root, .trash => return false,
            .node => |next| {
                if (std.meta.eql(next, node)) return true;
                cur = next;
            },
        }
    }
}

/// Create a new structural node under `parent`, at `order_key` among its
/// siblings (see `orderKeyBetween`). Returns the new node's id — also
/// usable immediately with `mapSet`/`mapGet` (see the section doc
/// comment). This write can never be cycle-rejected: a brand-new node
/// cannot already be anyone's ancestor.
///
/// `error.OrderKeyTooLong` if `order_key.len > max_order_key_len`: this is
/// DATA-DRIVEN (order-key growth is ~N/8 bytes under adversarial
/// same-locus reordering — `orderKeyBetween`'s doc comment — and periodic
/// sibling-key rebalancing isn't implemented yet, see `compact`'s doc
/// comment on structural ops), not a caller-contract bug, so it is a
/// real error a caller must be able to handle, in every build mode — NOT
/// an `assert` (which compiles out in `ReleaseFast`/`ReleaseSmall`,
/// which would silently reopen the exact un-round-trippable hole
/// `max_order_key_len`'s doc comment promises is closed).
pub fn structCreate(self: *ObjectDoc, gpa: Allocator, parent: StructRef, order_key: []const u8) (Allocator.Error || error{OrderKeyTooLong})!ObjId {
    if (order_key.len > max_order_key_len) return error.OrderKeyTooLong;
    const agent = self.agent.?;
    try self.ensureRoot(gpa);
    const key_str = try self.intern(gpa, order_key);
    const lv = try self.history.addLocal(gpa, agent, .{ .struct_create = .{ .parent = parent, .order_key = key_str } });
    try self.ensureNodeMap(gpa);
    _ = try self.makeValueNode(gpa, .new_map, lv);
    const id = self.history.idOf(lv);
    try self.struct_parents.put(gpa, id, .{
        .parent = toPlacementRef(parent),
        .order_key = key_str,
        .key_writer = id,
        .conflict_count = 1,
        .cycle_broken = false,
    });
    return id;
}

/// Identity-preserving move: one parent-register write against an
/// existing structural node. If this write would make `node` its own
/// ancestor, it is deterministically NOT applied to the materialized
/// tree (the event is still recorded — never un-appended, matching the
/// CRDT's no-true-deletes discipline — but `node`'s EFFECTIVE PARENT is
/// unchanged); a later `merge`'s full-history replay reconstructs the
/// identical outcome (see `wouldCycleLocal`'s doc comment), so the
/// EFFECTIVE PARENT this diverges from what a fresh replica opening the
/// same bytes would see. INVARIANT `struct_parents` must uphold even on
/// this refusal path: the stored entry must always equal the canonical
/// resolution of the CURRENT (post-append) history — not just "the
/// currently-accepted parent" — because `structConflictCount`/
/// `structCycleBroken` are documented as reporting THAT resolution. See
/// the refusal branch below for why the conflict metadata (not the
/// parent) still changes.
///
/// `error.OrderKeyTooLong` if `order_key.len > max_order_key_len` — see
/// `structCreate`'s doc comment for why this is a real, data-driven
/// error rather than an `assert`.
pub fn structMove(self: *ObjectDoc, gpa: Allocator, node: ObjId, parent: StructRef, order_key: []const u8) (Allocator.Error || error{OrderKeyTooLong})!void {
    if (order_key.len > max_order_key_len) return error.OrderKeyTooLong;
    const agent = self.agent.?;
    const key_str = try self.intern(gpa, order_key);
    const lv = try self.history.addLocal(gpa, agent, .{ .struct_move = .{ .node = node, .parent = parent, .order_key = key_str } });
    const id = self.history.idOf(lv);
    if (self.wouldCycleLocal(node, parent)) {
        // Refused: the EFFECTIVE parent doesn't change. But this write's
        // causal parents are the CURRENT frontier (`addLocal`), which
        // causally dominates every write this replica has ever seen —
        // so it supersedes every prior write to `node`'s register in the
        // canonical-order sense, exactly like
        // `objects_state.Walker.resolveStructs`'s `conflict_live`
        // bookkeeping (superseded entries are dropped WHETHER OR NOT the
        // superseding write itself goes on to be accepted). The
        // antichain therefore collapses to exactly {this refused write}
        // — size 1 — and the winner (whatever `node` already resolved to)
        // sits OUTSIDE it: `conflict_count = 1`, `cycle_broken = true`.
        // Leaving the old (pre-this-write) metadata in place here would
        // make this replica's OWN accessors disagree with what its own
        // history canonically resolves to — and nothing else ever
        // corrects it (`merge`'s `emitStructEffects` only emits an
        // effect when a FRESH before/after diff differs; a diff against
        // itself never does), so the divergence would be permanent, not
        // just transient.
        if (self.struct_parents.getPtr(node)) |cur| {
            cur.conflict_count = 1;
            cur.cycle_broken = true;
        }
        return;
    }
    try self.struct_parents.put(gpa, node, .{
        .parent = toPlacementRef(parent),
        .order_key = key_str,
        .key_writer = id,
        .conflict_count = 1,
        .cycle_broken = false,
    });
}

/// Sugar: move to `.trash` — "trash is another parent" (F3). Not
/// recursive: `node`'s children keep pointing at it; they simply become
/// unreachable from `.root` until `node` (with its whole subtree,
/// including any structural edits made to it while hidden) is moved back
/// out. See `structChildren(.trash)` / `structParent` to inspect trashed
/// nodes.
pub fn structDelete(self: *ObjectDoc, gpa: Allocator, node: ObjId) Allocator.Error!void {
    // The empty order key can never trip `error.OrderKeyTooLong` (`0 <=
    // max_order_key_len` always) — narrow back to `Allocator.Error` so
    // `structDelete`'s own callers don't have to handle an error that is
    // structurally unreachable for them, rather than widening this
    // signature to match `structMove`'s.
    self.structMove(gpa, node, .trash, &.{}) catch |err| switch (err) {
        error.OrderKeyTooLong => unreachable,
        else => |e| return e,
    };
}

// ── Structural reads ───────────────────────────────────────────────────

/// The effective parent of `node`'s parent register (F3, delta 6).
///
/// LANDMINE (`stemma-unification.md` §4.1 risk 1): this winner is picked
/// by a DIFFERENT rule than `ValueRef.mapGet`'s. `mapGet` picks the
/// greatest `(agent name, seq)` among the causally-maximal antichain — a
/// purely LOCAL, per-key rule that never needs to look outside one
/// register. A structural node's parent-register winner instead needs a
/// GLOBAL, replica-portable Lamport-then-(agent name, seq) canonical
/// order over EVERY structural write in the document, replayed once with
/// per-write cycle rejection — cross-node cycles (A moves under B while B
/// concurrently moves under A) cannot be resolved one register at a time.
/// In the common case this still lands on a causally-maximal
/// conflict-set member (see `structConflictCount`) — but when every
/// causally-dominant write to a node's register would cycle, the
/// effective winner falls back to an EARLIER, already-superseded write
/// (in the limit, the node's own `structCreate`) that sits OUTSIDE the
/// reported conflict set entirely. `structCycleBroken` reports exactly
/// this case. Never assume "greatest antichain member" for a structural
/// parent the way it is safe to for a map key — a structural editor or
/// projection surfacing "why is this node here" MUST check
/// `structCycleBroken` before explaining the placement as "the newest
/// concurrent write."
pub fn structParent(self: *const ObjectDoc, node: ObjId) ?StructRef {
    const p = self.struct_parents.get(node) orelse return null;
    return switch (p.parent) {
        .root => .root,
        .trash => .trash,
        .node => |id| .{ .node = id },
    };
}

/// Size of `node`'s parent-register conflict set (concurrent writes still
/// causally-maximal) — same shape as `ValueRef.mapConflictCount`. See
/// `structParent`'s doc comment: unlike a map key, the EFFECTIVE winner
/// can sit outside this set (`structCycleBroken`).
pub fn structConflictCount(self: *const ObjectDoc, node: ObjId) usize {
    const p = self.struct_parents.get(node) orelse return 0;
    return p.conflict_count;
}

/// True iff `node`'s effective parent (`structParent`) is NOT a member of
/// its own reported conflict set (`structConflictCount`) — the
/// cycle-break survivor case named in `structParent`'s doc comment. This
/// is the minimal honest query a projection needs to surface "a cycle
/// was broken, this node landed at a non-obvious parent" rather than
/// silently presenting the fallback parent as an ordinary uncontested
/// write.
pub fn structCycleBroken(self: *const ObjectDoc, node: ObjId) bool {
    const p = self.struct_parents.get(node) orelse return false;
    return p.cycle_broken;
}

/// `node`'s current order-key bytes (see `orderKeyBetween`) — the sort
/// key `structChildren` uses among siblings. Borrowed (valid for the
/// doc's lifetime, like `ValueRef.asStr`).
pub fn structOrderKey(self: *const ObjectDoc, node: ObjId) ?[]const u8 {
    const p = self.struct_parents.get(node) orelse return null;
    return self.str(p.order_key);
}

/// Children of `parent`, sorted by order key then authoring identity
/// (the tiebreak `orderKeyBetween`'s doc comment names — two replicas
/// computing the identical midpoint independently). Caller owns.
pub fn structChildren(self: *const ObjectDoc, gpa: Allocator, parent: StructRef) Allocator.Error![]ObjId {
    var out: std.ArrayList(ObjId) = .empty;
    errdefer out.deinit(gpa);
    const want = toPlacementRef(parent);
    var it = self.struct_parents.iterator();
    while (it.next()) |kv| {
        if (structRefEql(kv.value_ptr.parent, want)) try out.append(gpa, kv.key_ptr.*);
    }
    const SortCtx = struct {
        doc: *const ObjectDoc,
        fn less(ctx: @This(), a: EventId, b: EventId) bool {
            const sa = ctx.doc.struct_parents.get(a).?;
            const sb = ctx.doc.struct_parents.get(b).?;
            const order = std.mem.order(u8, ctx.doc.str(sa.order_key), ctx.doc.str(sb.order_key));
            if (order != .eq) return order == .lt;
            const na = ctx.doc.history.agentName(sa.key_writer.agent);
            const nb = ctx.doc.history.agentName(sb.key_writer.agent);
            const name_order = std.mem.order(u8, na, nb);
            if (name_order != .eq) return name_order == .lt;
            return sa.key_writer.seq < sb.key_writer.seq;
        }
    };
    std.mem.sort(EventId, out.items, SortCtx{ .doc = self }, SortCtx.less);
    return out.toOwnedSlice(gpa);
}

// ── Reads ───────────────────────────────────────────────────────────

pub fn root(self: *const ObjectDoc) ValueRef {
    return .{ .doc = self, .node = 0 };
}

/// A `ValueRef` for `obj` (`null` = root map) — the inverse of `ValueRef.
/// objId`. Every mutator (`mapSet`/`listInsert`/…) hands back an `ObjId`
/// naming a freshly created object with no way back into `ValueRef`
/// navigation short of re-walking from `root()`; callers that keep an
/// `ObjId` around (a graph facade's typed node handles, an identity
/// anchor) need to resolve it back to a readable value without that walk.
/// O(1): an `ObjId` is already validated causal identity, so this is a
/// lookup of its materialized tree node, same trust contract as every
/// mutator that takes `obj: ?ObjId` (caller passes an id of the right
/// kind; wrong-kind access hits the same `nodePtr()`-shaped assumptions
/// `ValueRef`'s accessors already make).
pub fn ref(self: *const ObjectDoc, obj: ?ObjId) ValueRef {
    return .{ .doc = self, .node = self.resolveObjNode(obj) };
}

pub const ValueRef = struct {
    doc: *const ObjectDoc,
    node: u32,

    fn nodePtr(self: ValueRef) ?*const Node {
        if (self.doc.nodes.items.len == 0) return null; // pristine doc
        return &self.doc.nodes.items[self.node];
    }

    pub fn kind(self: ValueRef) Kind {
        const n = self.nodePtr() orelse return .map;
        return switch (n.*) {
            .scalar => |s| switch (s) {
                .null_ => .null_,
                .bool_ => .bool_,
                .int => .int,
                .float => .float,
                .str => .str,
            },
            .map => .map,
            .list => .list,
            .text => .text,
        };
    }

    pub fn asBool(self: ValueRef) bool {
        return self.nodePtr().?.scalar.bool_;
    }
    pub fn asInt(self: ValueRef) i64 {
        return self.nodePtr().?.scalar.int;
    }
    pub fn asFloat(self: ValueRef) f64 {
        return self.nodePtr().?.scalar.float;
    }
    pub fn asStr(self: ValueRef) []const u8 {
        return self.doc.str(self.nodePtr().?.scalar.str);
    }
    pub fn textRope(self: ValueRef) *const Rope {
        return &self.nodePtr().?.text.rope;
    }

    /// The object's identity, usable for edits (`null` = root map).
    /// Portable and stable regardless of compaction — see `Node`'s doc
    /// comment.
    pub fn objId(self: ValueRef) ?ObjId {
        const n = self.nodePtr() orelse return null;
        return switch (n.*) {
            .map => |m| m.obj,
            .list => |l| l.obj,
            .text => |t| t.obj,
            .scalar => unreachable, // scalars have no identity
        };
    }

    fn slotOf(self: ValueRef, key: []const u8) ?*const TreeSlot {
        const n = self.nodePtr() orelse return null;
        for (n.map.slots.items) |*s| {
            if (std.mem.eql(u8, self.doc.str(s.key), key)) return s;
        }
        return null;
    }

    /// Deterministic winner among concurrent values: greatest
    /// (agent name, seq) of the setting event. Null if the key is
    /// absent/deleted.
    pub fn mapGet(self: ValueRef, key: []const u8) ?ValueRef {
        const slot = self.slotOf(key) orelse return null;
        if (slot.values.items.len == 0) return null;
        var best: TreeVal = slot.values.items[0];
        for (slot.values.items[1..]) |v| {
            if (self.doc.setOrder(best.set_id, v.set_id) == .lt) best = v;
        }
        return .{ .doc = self.doc, .node = best.node };
    }

    /// Number of concurrent values for `key` (>1 = a real conflict).
    pub fn mapConflictCount(self: ValueRef, key: []const u8) usize {
        const slot = self.slotOf(key) orelse return 0;
        return slot.values.items.len;
    }

    /// The i-th concurrent value (order deterministic but arbitrary).
    pub fn mapConflictAt(self: ValueRef, key: []const u8, i: usize) ValueRef {
        const slot = self.slotOf(key).?;
        return .{ .doc = self.doc, .node = slot.values.items[i].node };
    }

    pub fn listLen(self: ValueRef) usize {
        const n = self.nodePtr() orelse return 0;
        return n.list.elems.items.len;
    }

    pub fn listAt(self: ValueRef, i: usize) ValueRef {
        return .{ .doc = self.doc, .node = self.nodePtr().?.list.elems.items[i] };
    }

    pub const KeyIterator = struct {
        ref: ValueRef,
        i: usize = 0,

        pub fn next(self: *KeyIterator) ?[]const u8 {
            const n = self.ref.nodePtr() orelse return null;
            while (self.i < n.map.slots.items.len) {
                const s = &n.map.slots.items[self.i];
                self.i += 1;
                if (s.values.items.len > 0) return self.ref.doc.str(s.key);
            }
            return null;
        }
    };

    pub fn mapKeys(self: ValueRef) KeyIterator {
        return .{ .ref = self };
    }
};

/// Deterministic MV-register tiebreak: greatest `(agent name, seq)`.
/// Takes `EventId`s directly (not `Lv`s) — this must stay correct even
/// when `a`/`b` name a compacted-away `map_set` (see `Node`'s doc
/// comment); an agent's registration and its own `seq` numbering are
/// both stable across `compact`, so this needs nothing from `self.history`
/// beyond `agentName`, which is.
fn setOrder(self: *const ObjectDoc, a: EventId, b: EventId) std.math.Order {
    const name_order = std.mem.order(u8, self.history.agentName(a.agent), self.history.agentName(b.agent));
    if (name_order != .eq) return name_order;
    return std.math.order(a.seq, b.seq);
}

/// Canonical JSON dump (winner-only, keys sorted, text objects as
/// strings). Caller owns. Conflicts are invisible here — inspect them
/// via `mapConflictCount`. The structural tree (F3, delta 6) is ALSO
/// invisible here — a structural node only appears if something reached
/// it via ordinary map/list containment (`mapSet`/`listInsert`); its
/// `structParent`/`structChildren` placement is a separate forest this
/// walk never traverses. Inspect it via `structChildren(.root)` (and,
/// for trashed subtrees, `structChildren(.trash)`).
pub fn toJson(self: *const ObjectDoc, gpa: Allocator) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try self.dumpValue(gpa, &out, self.root());
    return out.toOwnedSlice(gpa);
}

fn dumpValue(self: *const ObjectDoc, gpa: Allocator, out: *std.ArrayList(u8), v: ValueRef) Allocator.Error!void {
    switch (v.kind()) {
        .null_ => try out.appendSlice(gpa, "null"),
        .bool_ => try out.appendSlice(gpa, if (v.asBool()) "true" else "false"),
        .int => try out.print(gpa, "{d}", .{v.asInt()}),
        .float => try out.print(gpa, "{d}", .{v.asFloat()}),
        .str => try dumpString(gpa, out, v.asStr()),
        .text => {
            const s = try v.textRope().toOwnedSlice(gpa);
            defer gpa.free(s);
            try dumpString(gpa, out, s);
        },
        .list => {
            try out.append(gpa, '[');
            for (0..v.listLen()) |i| {
                if (i > 0) try out.append(gpa, ',');
                try self.dumpValue(gpa, out, v.listAt(i));
            }
            try out.append(gpa, ']');
        },
        .map => {
            // Sorted keys for canonical output.
            var keys: std.ArrayList([]const u8) = .empty;
            defer keys.deinit(gpa);
            var it = v.mapKeys();
            while (it.next()) |k| try keys.append(gpa, k);
            std.mem.sort([]const u8, keys.items, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.order(u8, a, b) == .lt;
                }
            }.lt);
            try out.append(gpa, '{');
            for (keys.items, 0..) |k, i| {
                if (i > 0) try out.append(gpa, ',');
                try dumpString(gpa, out, k);
                try out.append(gpa, ':');
                try self.dumpValue(gpa, out, v.mapGet(k).?);
            }
            try out.append(gpa, '}');
        },
    }
}

fn dumpString(gpa: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    try out.append(gpa, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(gpa, "\\\""),
        '\\' => try out.appendSlice(gpa, "\\\\"),
        '\n' => try out.appendSlice(gpa, "\\n"),
        '\r' => try out.appendSlice(gpa, "\\r"),
        '\t' => try out.appendSlice(gpa, "\\t"),
        else => if (c < 0x20) {
            try out.print(gpa, "\\u{x:0>4}", .{c});
        } else {
            try out.append(gpa, c);
        },
    };
    try out.append(gpa, '"');
}

// ── Versions & sync (same token model as TextDoc) ───────────────────

pub fn version(self: *const ObjectDoc, gpa: Allocator) Allocator.Error![]u8 {
    return core.encodeVersion(gpa, &self.history);
}

fn decodeVersion(
    self: *const ObjectDoc,
    gpa: Allocator,
    token: []const u8,
    strict: bool,
    out: *std.ArrayList(Lv),
) MergeError!void {
    return core.decodeVersion(&self.history, gpa, token, strict, out);
}

pub fn compareVersions(self: *const ObjectDoc, gpa: Allocator, a_token: []const u8, b_token: []const u8) MergeError!VersionOrder {
    return core.compareVersions(gpa, &self.history, a_token, b_token);
}

pub fn eventsSince(self: *const ObjectDoc, gpa: Allocator, remote_version: []const u8) (Allocator.Error || error{Corrupt})![]u8 {
    var known: std.ArrayList(Lv) = .empty;
    defer known.deinit(gpa);
    self.decodeVersion(gpa, remote_version, false, &known) catch |e| switch (e) {
        error.MissingDependency => unreachable, // lenient mode
        // `decodeVersion` is `MergeError`-shaped (it shares the alias with
        // fns that DO need `Unrealized`), but this call site never
        // actually produces it: `core.decodeVersion` only ever returns
        // `Corrupt`/`MissingDependency`. Partial checkout deliberately
        // keeps `eventsSince` sync-capable while holey (`encodeEvents`
        // emits a version-only base — see `openPartial`'s wire-contract
        // doc comment) rather than widening this signature for an
        // unreachable arm.
        error.Unrealized => unreachable,
        else => |err| return err,
    };
    var missing = try self.history.missingFrom(gpa, known.items);
    defer missing.deinit(gpa);
    return self.encodeEvents(gpa, missing.items);
}

/// Wire-encode the events in `to_version`'s causal past that the holder
/// of `from_version` lacks — `eventsSince` bounded above by `to` instead
/// of the current head. Mirrors `TextDoc.eventsBetween` exactly (same
/// contract, same `causal.diff` shape): the incremental-sync primitive
/// for a consumer tracking a past state (persistence deltas, a
/// saved-file mirror — `Document.zig`'s `peerSyncTo`, `w7-rebase.md` §1).
/// Every entry of `to_version` must be stored here
/// (`error.MissingDependency`); unknown `from` entries are ignored.
pub fn eventsBetween(
    self: *const ObjectDoc,
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
    return self.encodeEvents(gpa, d.a_only.items);
}

/// The whole document as its event graph (plus per-object compacted
/// bases, if any) — the durable form. Same contract as
/// `TextDoc.serialize`: a document with ANY active hole cannot be
/// persisted (`error.Unrealized`) — realize every span first. Unlike
/// `eventsSince`/`eventsBetween` (which stay usable while holey via a
/// version-only base section — see `encodeEvents`'s wire-format doc
/// comment), `serialize` promises a SELF-CONTAINED snapshot; a
/// version-only base can't honor that promise for any peer, including
/// this replica reopening its own bytes later.
pub fn serialize(self: *const ObjectDoc, gpa: Allocator) (Allocator.Error || error{Unrealized})![]u8 {
    if (self.text_holes.count() != 0) return error.Unrealized;
    var missing = try self.history.missingFrom(gpa, &.{});
    defer missing.deinit(gpa);
    return self.encodeEvents(gpa, missing.items);
}

pub fn open(gpa: Allocator, bytes: []const u8) MergeError!ObjectDoc {
    var doc: ObjectDoc = .empty;
    errdefer doc.deinit(gpa);
    const changes = try doc.merge(gpa, bytes);
    gpa.free(changes);
    return doc;
}

// ── Bulk load ──────────────────────────────────────────────────────
// `w7-rebase.md` §1/§4 W7-1, `stemma-unification.md` §4 risk 4's sibling
// gap: `TextDoc.openFromContent`'s bulk-load shape, generalized past "the
// whole doc IS the text" (`ObjectDoc`'s root is always a map —
// `ensureRoot` — so the degenerate one-text-node doc this builds is
// `root.map -> key: text`, exactly the shape `stemma-unification.md`'s W7
// study names as "text is a degenerate graph doc").

/// Bulk-load `content` as a fresh document whose root map has exactly one
/// key, `key`, holding a text object initialized to `content` — O(content)
/// cost, ZERO `text_ins` events (a naive port would cost one event per
/// scalar, `mapSet` + `textInsert` in a loop; a large file would be
/// millions of events for zero reason). The text object's CREATION event
/// (the `map_set` that mints it) is a real, retained event — unlike
/// `TextDoc.openFromContent`'s fully synthetic base, `compact`'s own
/// contract never folds away a creating `map_set` (object identity needs
/// it to keep resolving — see `compact`'s doc comment) — so this
/// reproduces EXACTLY the state `mapSet(root, key, .text)` followed by
/// inserting `content` char-by-char and then `compact`ing to that one
/// event would leave behind: the same `text_bases`/`base_version`/
/// `base_head` shape `compact` already produces and the wire (v2,
/// `object_magic_v2`) already carries, not a third base representation.
///
/// Independent loads of identical bytes UNDER THE SAME `key` mint the
/// same synthetic agent name (content-AND-key-hash-derived — see below),
/// so they share a document root and can sync as replicas of one
/// document. `self.agent` is left unset, same as
/// `open`/`TextDoc.openFromContent`: call `setAgent` before any local
/// edit.
///
/// `key` is folded into the digest (length-prefixed, then `content`),
/// NOT just `content` alone as `TextDoc.openFromContent` folds (`TextDoc`
/// has no key — the whole doc IS the text, so content-only identity is
/// already unambiguous there). For `ObjectDoc` it is NOT optional: two
/// independent loads of identical `content` under DIFFERENT `key`s must
/// mint DIFFERENT roots. If the digest ignored `key`, both loads would
/// mint the SAME `{base-<hash>, seq 0}` creation event id carrying
/// DIFFERENT `map_set` payloads (different key) — `historyPhase` dedups
/// purely by id with no payload check, the two `base_version` tokens
/// would be byte-identical (so the "same base" sync guard in `merge`
/// passes), and `eventsSince` between them would emit zero events, so the
/// two documents would silently believe they are the same doc at the
/// same version while actually holding different keys — a probability-1
/// deterministic divergence within the feature's own intended usage
/// (loading the same bytes under two different field names), not the
/// accepted 2^-64 SHA-256 collision posture `TextDoc.openFromContent`
/// relies on for content-only identity. Folding `key` in makes
/// same-content/different-key loads mint different roots, so any attempt
/// to sync them is a loud, honest `error.MissingDependency` instead.
pub fn openFromContent(gpa: Allocator, content: []const u8, key: []const u8) (Allocator.Error || error{Corrupt})!ObjectDoc {
    if (!std.unicode.utf8ValidateSlice(content)) return error.Corrupt;
    const scalars = std.unicode.utf8CountCodepoints(content) catch unreachable;

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var key_len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &key_len_buf, key.len, .little);
    hasher.update(&key_len_buf);
    hasher.update(key);
    hasher.update(content);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    var name_buf: [5 + 32]u8 = undefined;
    @memcpy(name_buf[0..5], "base-");
    for (digest[0..16], 0..) |b, i| {
        _ = std.fmt.bufPrint(name_buf[5 + i * 2 ..][0..2], "{x:0>2}", .{b}) catch unreachable;
    }
    const name = name_buf[0..];

    var doc: ObjectDoc = .empty;
    errdefer doc.deinit(gpa);
    const aid = try doc.history.registerAgent(gpa, name);

    // One real, structural event — never a target of `compact`'s
    // text-only folding (see the doc comment above) — mints the text
    // object's identity. `self.agent` is set only for this one call.
    doc.agent = aid;
    const obj = (try doc.mapSet(gpa, null, key, .text)).?;
    doc.agent = null;

    // Seed the compacted base directly (block-scoped so the `errdefer`
    // retires the instant ownership transfers into `text_bases` — a
    // later failure below must not double-free it).
    {
        const bytes_owned = try gpa.dupe(u8, content);
        errdefer gpa.free(bytes_owned);
        try doc.text_bases.put(gpa, obj, .{ .bytes = bytes_owned, .scalars = scalars });
    }
    {
        var rope = try Rope.fromSlice(gpa, content);
        errdefer rope.deinit(gpa);
        const node_idx = doc.resolveObjNode(obj);
        doc.nodes.items[node_idx].text.rope = rope;
    }

    // The one retained event is the whole document's stable point — same
    // doc-wide `base_version`/`base_head` shape `compact` leaves behind.
    doc.base_version = try doc.version(gpa);
    doc.base_head = doc.history.idOf(doc.history.lvOf(obj).?);

    return doc;
}

// ── Partial checkout ────────────────────────────────────────────────
// `w7-rebase.md` §1/§4 W7-1's last gap, `stemma-unification.md` §4 risk
// 4's sibling to `TextDoc.openPartial`/`realizeBase` — a text object whose
// compacted base is only partially fetched: realized chunks carry
// content, holes carry (bytes, scalars) metadata, exactly like TextDoc's
// own partial checkout, scoped PER TEXT OBJECT (`text_holes`, alongside
// `text_bases`) rather than doc-wide, since ObjectDoc's compacted state
// already is per-object (unlike TextDoc, which has exactly one sequence).
// Rope-level placeholders reuse `Rope.fromUnrealized` unmodified (zero
// rope work); the FugueMax/SeqWalker base-placeholder machinery
// (`SeqWalker.initBase`, `objects_state.Walker.getSeqWalker`) is ALREADY
// agnostic to whether a base scalar is a hole or fully realized content —
// it only ever consults `TextBase.scalars` (a COUNT), never the bytes
// themselves — so neither needed a single change for this feature.
// EVERY byte↔scalar BOUNDARY crossing between `obj`'s rope (realized
// content only) and the object's GLOBAL scalar space (hole-inclusive,
// what the `SeqWalker` above actually indexes) DOES need hole-aware
// treatment, though — this is new, and every crossing needed it, not a
// subset: `textInsert`/`textDelete`'s local-edit preconditions above,
// `checkHoleConflicts`/`insByteOffset`/`delByteRange` below (`merge`'s
// effect application), AND — caught late, by review, after an initial
// claim that they "fell out for free" turned out to be false —
// `objectAnchorAt`/`resolveObjectAnchors` (identity anchors) below: an
// anchor taken at or resolved to a position at-or-after a hole silently
// landed on the wrong element without the same `holeScalarsThrough`/
// `insByteOffset` treatment. `TextDoc.anchorAt`/`resolveAnchors` carried
// the identical latent bug (this file's port target) and got the same
// fix. See each function's own doc comment for the mechanism.
//
// SCOPE: `openPartial` mirrors `openFromContent`'s shape exactly (one
// synthetic-identity text object under `key` in the root map) rather than
// the fully general "partial checkout of an arbitrary already-`compact`'d
// document with real multi-event pre-history" — that would need shipping
// the WHOLE retained non-text event graph up front (map/list/struct
// events never fold — see `compact`'s doc comment), which is exactly
// `merge`'s existing v2 bootstrap path, just with the text_base content
// made lazy; a materially larger feature than W7-1's actual need (weft's
// `Document` is ALWAYS exactly this one-root-map-one-text-key shape,
// `openFromContent`'s own degenerate case) and not attempted here. Given
// that, `base_version` still comes from the HOST (unlike
// `openFromContent`, which derives a synthetic one from a content hash —
// impossible here, since a partial checkout by definition doesn't have
// the pristine bytes of its holes to hash) — `agents` does too, for the
// SAME reason `TextDoc.openPartial` needs it: a real compacted document's
// agents may carry nonzero `seq_base` watermarks (an agent whose TEXT
// edits folded into this or an EARLIER object's base, even though their
// OWN `map_set`/`struct_create` events never do — see `compact`'s doc
// comment on why only text folds) that a fresh replica has no other way
// to learn.
//
// WIRE CONTRACT for a holey document (checked against TextDoc's ACTUAL
// behavior, then deliberately narrowed — see below): `serialize` refuses
// outright (`error.Unrealized`, same as `TextDoc.serialize`) — a durable
// snapshot needs every byte. `eventsSince`/`eventsBetween` stay usable
// while holey: `encodeEvents` (below) emits a THIRD wire version,
// `object_magic_v3`, whenever the sender has any active hole — same
// spirit as `TextDoc`'s RLE `flags: 2` ("version-only base"), scoped to
// ObjectDoc's plain v1/v2 (no format-selection parameter) shape: v3's
// base section carries `base_version` (so a peer sharing that exact base
// can still validate/merge normally) but ZERO `text_base` entries — NEVER
// enough to bootstrap a fresh replica, by construction (`merge`'s
// bootstrap check below gates on `dec.base.?.full`, decoded false only
// for v3). Old decoders reject v3 bytes outright (same one-directional
// "additive" story as every other wire delta here) — never emitted
// unless the sender itself is holey, so a doc that never partial-checked-
// out never produces v3 bytes at all. DELIBERATE DIVERGENCE from
// TextDoc: TextDoc additionally refuses `.unit` format specifically
// while holey (a caller-selectable format that always carries full base
// bytes); ObjectDoc has no such caller-selectable-format path to begin
// with (encoding always auto-selects v1/v2/v3 from document state), so
// there is no analogous escape hatch to close.

/// Per-object counterpart to `TextDoc.BaseHole`: one unrealized span of
/// a single text object's compacted base. `base_offset` is the fetch key
/// (byte offset in that object's pristine base — immutable); `cur_offset`
/// is where the span currently sits in that object's OWN rope (shifts
/// under edits to THAT object only — see `shiftHoles`).
pub const BaseHole = struct {
    base_offset: usize,
    cur_offset: usize,
    bytes: usize,
    scalars: usize,
};

/// Per-agent compaction watermark — same shape and purpose as
/// `TextDoc.AgentWatermark`; see the section doc comment above for why
/// `openPartial` needs a list of these rather than assuming just one.
pub const AgentWatermark = struct { name: []const u8, seq_base: u64 };

/// One span of a partial text-object base, in pristine-base order — same
/// shape as `TextDoc.BaseChunk`, scoped to ONE object's base instead of
/// the whole document.
pub const BaseChunk = union(enum) {
    /// Fetched content (UTF-8, copied).
    content: []const u8,
    /// Unfetched span of known size.
    hole: struct { bytes: usize, scalars: usize },
};

/// This document's watermarks for serving `openPartial` peers — a direct
/// port of `TextDoc.agentWatermarks`'s agent-table iteration (doc-wide:
/// ObjectDoc's agents live on the shared `history`, not per text object).
/// Names borrow from the document (valid until `deinit`); caller frees
/// the slice only. Same CAVEAT as `TextDoc.agentWatermarks`: `compact`
/// rebuilds `self.history` onto a brand new `Graph` (a fresh `names`
/// arena, the old one freed) — a `name` borrowed here and held across an
/// intervening `compact()` call dangles, it does not merely go stale.
/// Safe as used today (a single-threaded serve call reads, encodes, and
/// frees the returned slice within one synchronous call); a caller that
/// caches this across calls must re-fetch after every `compact`. The
/// serve side otherwise needs nothing beyond fields already exposed:
/// `base_version` and `text_bases.get(obj)` (`.bytes`/`.scalars`) are the
/// ObjectDoc analog of what `TextDoc`'s own `serveBase` reads directly
/// off `base_bytes`/`base_version` — a future `GraphCollab.serveBase`
/// chunks ONE object's `text_bases` entry the same way
/// `remote_fs.serveBase` chunks `TextDoc.base_bytes` today.
pub fn agentWatermarks(self: *const ObjectDoc, gpa: Allocator) Allocator.Error![]AgentWatermark {
    const out = try gpa.alloc(AgentWatermark, self.history.agents.items.len);
    for (out, self.history.agents.items, 0..) |*w, a, i| {
        w.* = .{
            .name = self.history.agentName(@enumFromInt(i)),
            .seq_base = a.seq_base,
        };
    }
    return out;
}

/// Bootstrap a partially realized replica of a compacted document: same
/// degenerate one-root-map-one-text-key shape as `openFromContent`, with
/// `chunks` allowed to carry holes (see the section doc comment above for
/// full scope). `base_version` (a single-head token) names this
/// document's whole `base_head` — the stable point `ObjectDoc.compact`
/// (or `openFromContent`'s born-compacted shape) was last taken at — and
/// `agents` supplies every watermark the host reports (`agentWatermarks`).
/// `error.Corrupt`: duplicate agent names, the named head's agent is not
/// among `agents`, the head's `seq` is newer than that agent's registered
/// `seq_base` could ever have produced (see below), or a malformed chunk.
/// Sync works immediately; content inside holes waits for `realizeBase`.
///
/// INVARIANT (the reasoning ported from `TextDoc.openPartial`'s own
/// check — `if (head.seq >= seq_base) return error.Corrupt`, i.e. it
/// REJECTS `head.seq >= seq_base` and so only ever ACCEPTS `head.seq <
/// seq_base` — see its doc comment for why): `head.seq` must name an
/// event `head_agent` has ALREADY produced as far as `seq_base` is
/// concerned, i.e. accept `head.seq <= seq_base`, reject `head.seq >
/// seq_base` — the latter would name an event `head_agent` hasn't gotten
/// to yet, which can never be a valid compaction boundary. The accepted
/// range is one wider than `TextDoc`'s (`<=`, not `<`) because `TextDoc`
/// never adds a single graph event for its base — `seq_base` alone fully
/// describes it, so `<` is the tight bound — while `ObjectDoc`'s base
/// ALSO needs one real, live event: the `map_set` that
/// mints `key`'s text object (never itself foldable — `compact`'s own
/// doc comment). Two regimes fall out of the same `<=`, not one:
///  - `head.seq == seq_base`: NOTHING of `head_agent`'s has ever folded
///    (the founder's own degenerate creation, `openFromContent`'s shape,
///    where `base_head` names that very `map_set`) — `head_agent` IS the
///    creator, and `head` names its creation event exactly.
///  - `head.seq < seq_base`: real editing happened and this doc was
///    `compact`ed at the resulting head, folding `head_agent`'s events up
///    through `head.seq` (`causal.compactGraph` raises `seq_base` to
///    `head.seq + 1` in exactly this case) — `head_id` now names a
///    FOLDED `text_ins`/`text_del`, not a creation event, and it can
///    never be re-added (`EventGraph.add`'s own precondition, `id.seq ==
///    nextSeq(id.agent)`, makes a lower-than-`seq_base` seq permanently
///    unreachable). `ObjectDoc.compact`'s per-agent-prefix guard (its own
///    doc comment: "a SINGLE agent that both creates a text object AND
///    later edits it... is error.NotCompactable") proves `head_agent`,
///    having folded anything at all, can therefore NEVER be `key`'s
///    creator — so in this regime the creator is a DIFFERENT agent, and
///    for this function's degenerate one-object scope (nothing else can
///    exist in the document before its one object is created, same as
///    `openFromContent`) that agent is unambiguously whichever one
///    registered FIRST: `agents[0]`.
pub fn openPartial(
    gpa: Allocator,
    base_version: []const u8,
    agents: []const AgentWatermark,
    key: []const u8,
    chunks: []const BaseChunk,
) (Allocator.Error || error{Corrupt})!ObjectDoc {
    var doc: ObjectDoc = .empty;
    errdefer doc.deinit(gpa);

    for (agents) |w| {
        const aid = try doc.history.registerAgent(gpa, w.name);
        doc.history.agents.items[@intFromEnum(aid)].seq_base = w.seq_base;
    }
    if (doc.history.agents.items.len != agents.len) return error.Corrupt; // duplicate names

    const head = try versionSingleEntry(base_version);
    const head_agent = doc.history.findAgent(head.name) orelse return error.Corrupt;
    const head_seq_base = doc.history.agents.items[@intFromEnum(head_agent)].seq_base;
    if (head.seq > head_seq_base) {
        return error.Corrupt; // names an event `head_agent` hasn't produced
    }
    const founder_creation = head.seq == head_seq_base;
    const head_id: EventId = .{ .agent = head_agent, .seq = head.seq };

    // The one retained event: exactly `openFromContent`'s `mapSet` call.
    // `founder_creation` (see the doc comment above): its (agent, seq) is
    // fixed by `head_id` — `head_agent`'s freshly-registered `lv_by_seq`
    // is empty, so `addLocal`'s `nextSeq` computes exactly `seq_base`,
    // which the check above already pinned to `head.seq`. Otherwise
    // (`head_id` names a folded, non-creation event — unreachable via
    // `addLocal` at all) the creator is `agents[0]` instead.
    const creator: AgentId = if (founder_creation) head_agent else @enumFromInt(0);
    // Loud, not silent: `compact`'s per-agent-prefix guard (see the doc
    // comment above) PROVES `head_agent` can never be `agents[0]` once
    // it's folded anything — if a caller's `base_version`/`agents` pair
    // violates that (a self-inconsistent or malicious combination), mint
    // the wrong identity under a match here anyway would be a silent
    // correctness bug (a partial replica quietly diverging from the real
    // document's object identity) rather than a crash. Crash.
    if (!founder_creation) assert(@intFromEnum(head_agent) != @intFromEnum(creator));
    doc.agent = creator;
    const obj = (try doc.mapSet(gpa, null, key, .text)).?;
    doc.agent = null;
    if (founder_creation) assert(std.meta.eql(obj, head_id));

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

    // Ownership of `base` transfers to `doc.text_bases` immediately (even
    // though it's still all-zero at this point) — mirrors
    // `TextDoc.openPartial`'s "assign to the field right after
    // allocating" discipline: any later failure in the chunk loop below
    // is then covered by the ALREADY-ARMED `errdefer doc.deinit(gpa)`
    // above, which frees every `text_bases` value unconditionally.
    const base = try gpa.alloc(u8, total_bytes);
    @memset(base, 0);
    try doc.text_bases.put(gpa, obj, .{ .bytes = base, .scalars = total_scalars });

    const node_idx = doc.resolveObjNode(obj);
    var offset: usize = 0;
    for (chunks) |c| switch (c) {
        .content => |bytes| {
            if (bytes.len > 0) {
                @memcpy(base[offset..][0..bytes.len], bytes);
                var piece = try Rope.fromSlice(gpa, bytes);
                defer piece.deinit(gpa);
                try doc.nodes.items[node_idx].text.rope.append(gpa, &piece);
            }
            offset += bytes.len;
        },
        .hole => |h| {
            const gop = try doc.text_holes.getOrPut(gpa, obj);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(gpa, .{
                .base_offset = offset,
                .cur_offset = offset,
                .bytes = h.bytes,
                .scalars = h.scalars,
            });
            var piece = try Rope.fromUnrealized(gpa, h.bytes);
            defer piece.deinit(gpa);
            try doc.nodes.items[node_idx].text.rope.append(gpa, &piece);
            offset += h.bytes;
        },
    };

    doc.base_version = try gpa.dupe(u8, base_version);
    doc.base_head = head_id;

    return doc;
}

/// Whether text object `obj`'s base is fully realized. Always true for an
/// object that never went through `openPartial` (including one with no
/// compacted base at all).
pub fn baseRealized(self: *const ObjectDoc, obj: ObjId) bool {
    const holes = self.text_holes.get(obj) orelse return true;
    return holes.items.len == 0;
}

/// The fetch list for text object `obj`: its unrealized base spans,
/// `base_offset` being the byte range key in ITS pristine base. Borrows
/// from the document. Empty for an object with no active holes.
pub fn unrealizedBase(self: *const ObjectDoc, obj: ObjId) []const BaseHole {
    const holes = self.text_holes.get(obj) orelse return &.{};
    return holes.items;
}

/// Supply the pristine content of one unrealized span of text object
/// `obj`'s base (whole-span; identified by `base_offset`). Verified
/// against the recorded byte and scalar counts (`error.Corrupt` on
/// mismatch, or if `obj` has no matching hole — nothing changes). Not an
/// edit: offsets, anchors, and versions are all unaffected. Same contract
/// as `TextDoc.realizeBase`, scoped to `obj`.
pub fn realizeBase(
    self: *ObjectDoc,
    gpa: Allocator,
    obj: ObjId,
    base_offset: usize,
    content: []const u8,
) (Allocator.Error || error{Corrupt})!void {
    const holes = self.text_holes.getPtr(obj) orelse return error.Corrupt;
    const idx = for (holes.items, 0..) |h, i| {
        if (h.base_offset == base_offset) break i;
    } else return error.Corrupt;
    const h = holes.items[idx];
    if (content.len != h.bytes) return error.Corrupt;
    if (!std.unicode.utf8ValidateSlice(content)) return error.Corrupt;
    if ((std.unicode.utf8CountCodepoints(content) catch unreachable) != h.scalars) return error.Corrupt;
    const node_idx = self.resolveObjNode(obj);
    try self.nodes.items[node_idx].text.rope.realize(gpa, h.cur_offset, content);
    const base = self.text_bases.getPtr(obj).?;
    // `jw.TextBase.bytes` is `[]const u8` (every OTHER writer treats a
    // base as immutable once built); `realizeBase` is the one place that
    // legitimately mutates it in place — we exclusively own this
    // allocation (nothing else ever aliases a `text_bases` entry's bytes).
    @memcpy(@constCast(base.bytes)[h.base_offset..][0..h.bytes], content);
    _ = holes.orderedRemove(idx);
    if (holes.items.len == 0) {
        var removed = self.text_holes.fetchRemove(obj).?;
        removed.value.deinit(gpa);
    }
}

// ── Merge ───────────────────────────────────────────────────────────

/// Integrate encoded remote events; returns the change stream (caller
/// owns). Same atomic-reject semantics as `TextDoc.merge`.
///
/// Sync-across-compaction discipline, ported from `TextDoc.merge`: a
/// batch carrying a base section (§3 step 4's v2 wire — see `compact`)
/// can only be accepted by a replica that is EITHER empty (bootstraps,
/// adopting the batch's base) OR already compacted to the EXACT SAME
/// stable point (`base_version` equality) — anything else, including a
/// non-empty uncompacted replica, or one compacted to a DIFFERENT point,
/// is rejected whole with `error.MissingDependency`. This is the "peer
/// behind the compaction point" case named in the compaction study: LOUD
/// and documented, never a silent partial/divergent merge. A peer that IS
/// caught up to (or past) the stable point converges normally — its own
/// new events' parent references resolve either directly or (for the
/// exact boundary event) implicitly via `batch_head`, same as
/// `TextDoc.historyPhase`'s "validated to be the base head" comment.
/// Full catch-up FROM an arbitrary earlier point across a compaction
/// boundary (an uncompacted or differently-compacted peer automatically
/// re-synchronizing) is explicitly not attempted here — a materially
/// larger feature than this step carries; see `TextDoc.openPartial`'s
/// watermark/chunk machinery for what that would need, ported wholesale.
pub fn merge(self: *ObjectDoc, gpa: Allocator, bytes: []const u8) MergeError![]Change {
    var dec = try Decoder.init(gpa, bytes);
    defer dec.deinit(gpa);

    // A version-only base section (wire v3 — a holey sender, see
    // `openPartial`'s wire-contract doc comment) can never seed a
    // bootstrap: it carries no text content to bootstrap FROM. Mirrors
    // `TextDoc.merge`'s `dec.base.?.bytes != null` guard.
    const bootstrap = dec.base != null and dec.base.?.full and
        self.base_version.len == 0 and self.history.eventCount() == 0;
    if (dec.base) |b| {
        if (!bootstrap and !std.mem.eql(u8, self.base_version, b.version)) {
            return error.MissingDependency;
        }
    }

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

    // The base boundary an incoming compacted-parent reference (or, for a
    // bootstrap, a text-base entry) may name implicitly.
    var batch_head: ?EventId = self.base_head;
    if (bootstrap) {
        const entry = try versionSingleEntry(dec.base.?.version);
        const agent = self.history.findAgent(entry.name) orelse return error.Corrupt;
        batch_head = .{ .agent = agent, .seq = entry.seq };
    }

    try dec.validate(gpa, self, aids, eff_base, batch_head);

    // Commit the bootstrap watermarks now (validated) — event-adding
    // below needs `nextSeq` to already reflect them.
    if (bootstrap) {
        for (aids, eff_base) |aid, eff| {
            self.history.agents.items[@intFromEnum(aid)].seq_base = eff;
        }
    }

    // Adopt the bootstrap's per-object base METADATA (bytes + scalar
    // count) into `self.text_bases` BEFORE `historyPhase` runs — not
    // just before its own effect fires. `historyPhase`'s replay
    // (`objects_state.Walker.getSeqWalker`) seeds a text object's
    // FugueMax placeholder items from `self.text_bases` the FIRST time
    // replay touches that object, and it's replay that computes every
    // `text_ins`/`text_del` EFFECT's position relative to that
    // placeholder-seeded sequence. A base adopted only afterward (this
    // function's older bug: it moved the ROPE seed to the object's
    // creation effect but left the METADATA seed there too) makes
    // replay treat the object as having ZERO base scalars — for a
    // TRIVIAL (empty) base this happens to be harmless (there's nothing
    // to anchor into), but for a REAL, non-empty base it corrupts every
    // position computed against it, surfacing as `error.Corrupt` deep in
    // `Sequence.applyInsert`/`applyDelete`, not a content clobber.
    // `self.text_bases` is necessarily empty before a true bootstrap (no
    // events yet ⇒ no objects yet ⇒ no bases), so a rejected batch must
    // remove exactly the entries THIS call adds — `seeded_bases` tracks
    // them and the `errdefer` below unwinds them on ANY later failure in
    // this function (`historyPhase`, tree application, allocation),
    // same "leave `self` untouched on rejection" discipline as every
    // other error path here. Rope materialization stays where it was:
    // at the object's creation effect, in the tree-application loop
    // below (`adoptTextBaseRope`) — the node (and `obj_index` entry)
    // only exists once `makeValueNode` has run for it, which happens
    // there, not in `historyPhase`.
    var seeded_bases: std.ArrayList(EventId) = .empty;
    defer seeded_bases.deinit(gpa);
    errdefer for (seeded_bases.items) |id| {
        if (self.text_bases.fetchRemove(id)) |kv| gpa.free(kv.value.bytes);
    };
    if (bootstrap) {
        for (dec.base.?.text_bases) |tb| {
            const id: EventId = .{ .agent = aids[tb.obj.agent_idx], .seq = tb.obj.seq };
            // Record intent BEFORE mutating `self.text_bases`, not after:
            // if `adoptTextBaseMetadata` itself fails partway (its own
            // `gpa.dupe`/`getOrPut` calls), it's guaranteed to have left
            // `self.text_bases` untouched for `id` (traced in its own
            // doc comment) — so an `id` appended here that never actually
            // got adopted is harmless (the `errdefer` above's
            // `fetchRemove` simply misses). The REVERSE order is the
            // real bug: appending AFTER a successful adopt leaves a
            // window where `self.text_bases` already has the entry but
            // `seeded_bases` doesn't yet know about it — an OOM on the
            // append itself would then orphan it, invisible to the
            // rollback (caught by this file's own OOM battery).
            try seeded_bases.append(gpa, id);
            try self.adoptTextBaseMetadata(gpa, id, tb);
        }
    }

    // Graph phase with wholesale rollback (including interned strings).
    var effects: std.ArrayList(Effect) = .empty;
    defer effects.deinit(gpa);
    const any_new = try self.historyPhase(gpa, &dec, aids, &effects);

    // A bootstrap's base fields (`base_version`/`base_head`) commit
    // regardless of `any_new` (a compacted-but-otherwise-idle doc's
    // serialize() can legitimately carry zero pending events).
    if (bootstrap) {
        gpa.free(self.base_version);
        self.base_version = try gpa.dupe(u8, dec.base.?.version);
        self.base_head = batch_head;
    }

    if (!any_new) {
        // Base METADATA is already adopted (above); there is no ROPE to
        // seed here — zero new events means the tree-application loop
        // below never runs, so no node exists yet for any object this
        // batch's base section could name (a fresh bootstrap's base
        // section always names an object created by SOME event in the
        // very same batch — see the tree-application loop's own comment
        // on this — so a batch reaching this branch has no text bases to
        // begin with; `seeded_bases` is empty here in practice).
        return try gpa.alloc(Change, 0);
    }

    // Tree application + change stream.
    try self.ensureRoot(gpa);
    try self.ensureNodeMap(gpa);
    var changes: std.ArrayList(Change) = .empty;
    errdefer changes.deinit(gpa);
    var buf: [4]u8 = undefined;
    for (effects.items) |eff| {
        switch (eff) {
            .map_add => |e| {
                const map_node = self.nodeOfObjLv(e.obj orelse root_key);
                const value_node = try self.makeValueNode(gpa, e.val, e.set_lv);
                // `ADDPEER_BOOTSTRAP_HAZARD` — see `adoptTextBaseRope`'s
                // doc comment: materialize a freshly-created text
                // object's rope from its (already `merge`-seeded, see
                // above) base bytes HERE, right after its creation
                // effect, so any LATER effect in this same batch
                // targeting it (`.text_ins`/`.text_del`, necessarily
                // later in causal/Lv order) lands on the seeded rope
                // instead of an empty one.
                if (bootstrap and e.val == .new_text) {
                    const id = self.history.idOf(e.set_lv);
                    if (self.text_bases.contains(id)) try self.adoptTextBaseRope(gpa, id);
                }
                const slot = try self.treeSlot(gpa, map_node, e.key);
                try slot.values.append(gpa, .{ .set_id = self.history.idOf(e.set_lv), .node = value_node });
                try changes.append(gpa, .{ .map = .{ .obj = self.objIdOf(e.obj), .key = self.str(e.key) } });
            },
            .map_remove => |e| {
                const map_node = self.nodeOfObjLv(e.obj orelse root_key);
                const slot = try self.treeSlot(gpa, map_node, e.key);
                const target = self.history.idOf(e.set_lv);
                for (slot.values.items, 0..) |v, i| {
                    if (v.set_id.agent == target.agent and v.set_id.seq == target.seq) {
                        _ = slot.values.orderedRemove(i);
                        break;
                    }
                }
                try changes.append(gpa, .{ .map = .{ .obj = self.objIdOf(e.obj), .key = self.str(e.key) } });
            },
            .list_ins => |e| {
                const list_node = self.nodeOfObjLv(e.obj);
                const value_node = try self.makeValueNode(gpa, e.val, e.lv);
                // See the identical hook in `.map_add` above — a text
                // object can equally be created as a list element.
                if (bootstrap and e.val == .new_text) {
                    const id = self.history.idOf(e.lv);
                    if (self.text_bases.contains(id)) try self.adoptTextBaseRope(gpa, id);
                }
                try self.nodes.items[list_node].list.elems.insert(gpa, @intCast(e.index), value_node);
                try changes.append(gpa, .{ .list_ins = .{ .obj = self.history.idOf(e.obj), .index = @intCast(e.index) } });
            },
            .list_del => |e| {
                const list_node = self.nodeOfObjLv(e.obj);
                _ = self.nodes.items[list_node].list.elems.orderedRemove(@intCast(e.index));
                try changes.append(gpa, .{ .list_del = .{ .obj = self.history.idOf(e.obj), .index = @intCast(e.index) } });
            },
            .text_ins => |e| {
                const obj_id = self.history.idOf(e.obj);
                const holey = self.text_holes.contains(obj_id);
                const text_node = self.nodeOfObjLv(e.obj);
                const t = &self.nodes.items[text_node].text;
                const off = if (holey) self.insByteOffset(obj_id, e.pos) else t.rope.scalarToOffset(e.pos);
                const len = std.unicode.utf8Encode(e.ch, &buf) catch unreachable;
                _ = try self.nodes.items[text_node].text.rope.insert(gpa, off, buf[0..len]);
                // Keep `text_holes[obj_id]`'s `cur_offset` bookkeeping in
                // sync with a REMOTE effect landing near (never inside —
                // `checkHoleConflicts` already proved that) a hole, same
                // as the local-edit path (`textInsert`) already does —
                // mirrors `TextDoc.merge`'s `if (holey) self.shiftHoles(...)`.
                if (holey) self.shiftHoles(obj_id, off, len, 0);
                try appendTextChange(gpa, &changes, obj_id, .{ .offset = off, .removed = 0, .inserted = len });
            },
            .text_del => |e| {
                const obj_id = self.history.idOf(e.obj);
                const holey = self.text_holes.contains(obj_id);
                const text_node = self.nodeOfObjLv(e.obj);
                const t = &self.nodes.items[text_node].text;
                const range = if (holey) self.delByteRange(obj_id, e.pos) else Range{
                    .start = t.rope.scalarToOffset(e.pos),
                    .end = t.rope.scalarToOffset(e.pos + 1),
                };
                const start = range.start;
                const end = range.end;
                _ = try self.nodes.items[text_node].text.rope.delete(gpa, range);
                if (holey) self.shiftHoles(obj_id, start, 0, end - start);
                try appendTextChange(gpa, &changes, obj_id, .{ .offset = start, .removed = end - start, .inserted = 0 });
            },
            .struct_created => |e| {
                // Eager, in Lv order — see the `Effect.struct_created`
                // doc comment for why this can't wait for the deferred
                // `struct_parent` post-pass. Idempotent defensively (a
                // node is only ever created once, but `obj_index` is the
                // authoritative check, matching every other `.new_map`
                // site).
                const node_id = self.history.idOf(e.node);
                if (self.obj_index.get(node_id) == null) {
                    _ = try self.makeValueNode(gpa, .new_map, e.node);
                }
            },
            .struct_parent => |e| {
                const node_id = self.history.idOf(e.node);
                // Defensive fallback (should be unreachable: `struct_created`
                // always precedes this for the SAME node, since its Lv is
                // strictly lower — see `resolveObj`'s `.struct_create` case
                // for why creation always causally precedes any reference).
                if (self.obj_index.get(node_id) == null) {
                    _ = try self.makeValueNode(gpa, .new_map, e.node);
                }
                const parent_ref: StructPlacementRef = switch (e.parent) {
                    .root => .root,
                    .trash => .trash,
                    .node => |lv| .{ .node = self.history.idOf(lv) },
                };
                try self.struct_parents.put(gpa, node_id, .{
                    .parent = parent_ref,
                    .order_key = e.order_key,
                    .key_writer = self.history.idOf(e.key_writer),
                    .conflict_count = e.conflict_count,
                    .cycle_broken = e.cycle_broken,
                });
                try changes.append(gpa, .{ .structure = .{ .node = node_id } });
            },
        }
    }
    // No deferred rope-adoption sweep here: every `base.text_bases`
    // entry names an object created by SOME `.map_add`/`.list_ins`
    // effect in the loop just run (`Decoder.validate` confirmed each
    // entry's creating event is known or present in this very batch;
    // "known" is impossible on a true bootstrap, since `self` started
    // entirely empty) — the inline `adoptTextBaseRope` hooks in those
    // two effect arms above already materialized every one of them,
    // each right at its own creation, which is what makes any subsequent
    // `.text_ins`/`.text_del` for that object in this same loop land on
    // the seeded rope instead of clobbering it afterward
    // (`ADDPEER_BOOTSTRAP_HAZARD`). The METADATA half of every entry was
    // already adopted before `historyPhase` ran at all (see above) —
    // this loop only ever does the ROPE half.
    return changes.toOwnedSlice(gpa);
}

fn objIdOf(self: *const ObjectDoc, obj: ?Lv) ?ObjId {
    return if (obj) |lv| self.history.idOf(lv) else null;
}

fn appendTextChange(gpa: Allocator, changes: *std.ArrayList(Change), obj: ObjId, edit: Edit) Allocator.Error!void {
    if (changes.items.len > 0) {
        const last = &changes.items[changes.items.len - 1];
        if (last.* == .text and last.text.obj.agent == obj.agent and last.text.obj.seq == obj.seq) {
            const le = &last.text.edit;
            if (le.removed == 0 and edit.removed == 0 and edit.offset == le.offset + le.inserted) {
                le.inserted += edit.inserted;
                return;
            }
            if (le.inserted == 0 and edit.inserted == 0 and edit.offset == le.offset) {
                le.removed += edit.removed;
                return;
            }
        }
    }
    try changes.append(gpa, .{ .text = .{ .obj = obj, .edit = edit } });
}

// ── Identity anchors ────────────────────────────────────────────────
// Text objects only (list anchors would need an element-index rather than
// byte-offset shape; the study that scoped this delta names it as falling
// out "for free" if wanted, but does not require it — not built here).

/// `resolveObjectAnchors`'s `ctx`: an anchor is valid for `obj` iff its
/// event is a `.text_ins` targeting `obj` specifically — `ObjectDoc`'s
/// graph is a tree shared by every object, so (unlike `TextDoc`, where
/// every event is the one sequence's) the op tag alone isn't enough.
const TextInsertCtx = struct {
    graph: *const Graph,
    obj: Lv,

    pub fn isInsert(self: @This(), op: ObjectOp) bool {
        return switch (op) {
            .text_ins => |x| (self.graph.lvOf(x.obj) orelse return false) == self.obj,
            else => false,
        };
    }
};

/// Full silent replay of the whole graph (current state), handing back
/// only `obj`'s own `SeqWalker` — every other object's/register's replay
/// state is discarded. Walking the WHOLE graph (not just `obj`'s events)
/// is required for FugueMax correctness, not a missed optimization: an
/// event's insert/delete position is relative to the prepare state at
/// ITS OWN causal frontier, which `Walker.replayAll`'s retreat/advance
/// discipline (`seq_walker.movePrepareTo`) computes in one global Lv-order
/// pass — there is no cheaper per-object walk to do instead. Same cost
/// profile as `TextDoc.silentReplay` (also an unscoped, throwaway
/// per-call walker — a candidate for a persistent per-object cache, left
/// for later, same as TextDoc's `merge_walk` is for its own single
/// sequence).
fn silentObjectReplay(self: *const ObjectDoc, gpa: Allocator, obj_lv: Lv) Allocator.Error!SeqWalker {
    var w = Walker.initWithBases(&self.history, self.strings.items, &self.text_bases);
    defer w.deinit(gpa);
    var sink: std.ArrayList(Effect) = .empty;
    defer sink.deinit(gpa);
    const n: Lv = @intCast(self.history.eventCount());
    w.replayAll(gpa, n, &sink, null) catch |e| switch (e) {
        error.Corrupt => unreachable, // trusted local history
        else => |err| return err,
    };
    assert(sink.items.len == 0); // first_new == n: nothing emits
    // Detach `obj`'s walker before `w.deinit` below frees every object's
    // and register's state — an object with zero ins/del events never
    // got a `seqs` entry (`getSeqWalker` is only reached from inside an
    // actual ins/del dispatch). Two reasons that can happen: a freshly
    // created, still-empty text node — OR a text object whose ENTIRE
    // content has been compacted away (every retained event elsewhere in
    // the graph, none touching this object at all) — the base placeholder
    // never gets a chance to seed via `getSeqWalker` either, so build it
    // directly from `self.text_bases` here.
    if (w.seqs.fetchRemove(obj_lv)) |kv| return kv.value;
    var sw: SeqWalker = .empty;
    if (self.text_bases.get(self.history.idOf(obj_lv))) |base| {
        try sw.initBase(gpa, base.scalars, seq_walker.base_placeholder_lv);
    }
    return sw;
}

/// An identity anchor for the position `byte_offset` inside text object
/// `obj`. Same contract as `TextDoc.anchorAt` (`stickiness` picks
/// `.before`/`.after`); the returned `agent` slice is gpa-owned.
/// `error.Compacted` is part of the shared error set both as an honest
/// reservation for `ObjectDoc`'s own eventual compaction (not yet built)
/// AND the real, reachable answer for a `byte_offset` landing on `obj`'s
/// still-unrealized base content (partial checkout, `openPartial`) — see
/// `seq_walker.base_placeholder_lv`.
pub fn objectAnchorAt(
    self: *const ObjectDoc,
    gpa: Allocator,
    obj: ObjId,
    byte_offset: usize,
    stickiness: AnchorSide,
) AnchorError!EventAnchor {
    const node_idx = self.resolveObjNode(obj);
    assert(self.nodes.items[node_idx] == .text);
    const rope = &self.nodes.items[node_idx].text.rope;
    // Global scalar space (what the `SeqWalker` below indexes into)
    // includes `obj`'s unrealized base-hole scalars, which `rope`'s own
    // metric does NOT count (a hole leaf's summary carries zero scalars —
    // see `Rope`'s "Lazy / unrealized content" section) — `holeScalarsThrough`
    // compensates, exactly mirroring `textInsert`'s own byte→scalar step.
    // A no-op (adds 0) for the overwhelming common case of an object with
    // no active holes — this was the found bug: every OTHER byte↔scalar
    // boundary here (`textInsert`/`textDelete`, `merge`'s
    // `insByteOffset`/`delByteRange`) already got this treatment; the
    // anchor path silently didn't, so an anchor taken at or after a hole
    // resolved against the wrong element (interior to the hole's own
    // placeholder block instead of the real content past it).
    const scalar = rope.offsetToScalar(byte_offset) + self.holeScalarsThrough(obj, byte_offset);
    const total = rope.scalarLen() + self.holeScalarsThrough(obj, rope.byteLen());
    switch (stickiness) {
        .before => if (scalar == total) return .{ .agent = "", .side = .after },
        .after => if (scalar == 0) return .{ .agent = "", .side = .before },
    }
    const target_index = switch (stickiness) {
        .before => scalar,
        .after => scalar - 1,
    };

    const obj_lv = self.history.lvOf(obj).?;
    var sw = try self.silentObjectReplay(gpa, obj_lv);
    defer sw.deinit(gpa);
    return seq_walker.anchorAt(gpa, &self.history, &sw, target_index, stickiness);
}

/// Resolve identity anchors into text object `obj` to current byte
/// offsets. Deleted targets collapse to their deletion point. An anchor
/// naming an event outside `obj` (wrong object, or not a text insert at
/// all) is `error.Corrupt` — see `TextInsertCtx`. Same batching contract
/// as `TextDoc.resolveAnchors`: one replay amortized over the whole set.
pub fn resolveObjectAnchors(
    self: *const ObjectDoc,
    gpa: Allocator,
    obj: ObjId,
    anchors: []const EventAnchor,
    out: []usize,
) AnchorError!void {
    assert(anchors.len == out.len);
    const node_idx = self.resolveObjNode(obj);
    assert(self.nodes.items[node_idx] == .text);

    const obj_lv = self.history.lvOf(obj).?;
    var sw = try self.silentObjectReplay(gpa, obj_lv);
    defer sw.deinit(gpa);
    const ctx: TextInsertCtx = .{ .graph = &self.history, .obj = obj_lv };
    try seq_walker.resolveAnchors(gpa, &self.history, &sw, ctx, anchors, out);
    // The shared layer returns GLOBAL SCALAR positions (base-hole-
    // inclusive, same space `objectAnchorAt` targets above) — map to
    // bytes with the same hole-aware boundary mapping `merge`'s
    // insert-effect path uses (`insByteOffset`): a resolved identity's
    // item can only ever sit before or after `obj`'s hole block, never
    // interior to one (no edit is ever accepted interior to a hole —
    // `assertOutsideHoles`/`checkHoleConflicts` both refuse that), so the
    // insertion-point mapping is exactly the boundary mapping a READ
    // needs too. A no-op (degenerates to plain `scalarToOffset`) when
    // `obj` has no active holes — this was the other half of the found
    // bug (see `objectAnchorAt`'s doc comment): the old direct
    // `rope.scalarToOffset(o.*)` silently undercounted by every
    // preceding hole's scalars whenever one existed.
    for (out) |*o| o.* = self.insByteOffset(obj, @intCast(o.*));
}

// ── Time travel ─────────────────────────────────────────────────────

/// Materialize text object `obj`'s content as it was at `version_token`
/// — the per-object analog of `TextDoc.materializeAt` (a graph doc has no
/// single "the text" to time-travel through; `obj` names which text
/// object — `w7-rebase.md` §1's `materializeAt` row / `Document.textAt`).
/// `version_token` must be fully known to us, same contract as every
/// other version token here (`error.MissingDependency` otherwise).
/// `error.Corrupt` if `obj` had not yet been created as of that version
/// — there is no "the text" to travel to before the object exists. This
/// reuses `error.Corrupt` for a DIFFERENT meaning than its usual "the
/// bytes/token are malformed" sense elsewhere in this file (matching the
/// local convention `openPartial`/`realizeBase` already set for a
/// similar "structurally valid but not applicable here" case) — the
/// token itself is perfectly valid, it just names a point in this
/// object's history that predates the object's own existence. Named here
/// explicitly so a future caller does not read `error.Corrupt` as "bad
/// wire bytes" and treat it as fatal-and-unexpected rather than a normal,
/// checkable precondition failure.
/// Returns a fresh Rope the caller owns. O(history) replay, reusing the
/// exact same `Walker`/`text_bases` machinery `compact`'s own
/// `materializeTextBasesAt` drives (this is that operation without the
/// graph-rebuild commit, scoped to one object instead of every touched
/// one).
pub fn materializeAt(
    self: *const ObjectDoc,
    gpa: Allocator,
    obj: ObjId,
    version_token: []const u8,
) MergeError!Rope {
    var heads: std.ArrayList(Lv) = .empty;
    defer heads.deinit(gpa);
    try self.decodeVersion(gpa, version_token, true, &heads);

    const n: Lv = @intCast(self.history.eventCount());
    const include = try gpa.alloc(bool, n);
    defer gpa.free(include);
    @memset(include, false);
    var d = try self.history.diff(gpa, heads.items, &.{});
    defer d.deinit(gpa);
    for (d.a_only.items) |lv| include[lv] = true;

    const obj_lv = self.history.lvOf(obj).?;
    if (!include[obj_lv]) return error.Corrupt; // not created yet at this version
    assert(self.objKind(obj_lv) == .text);
    // Needs full base content — mirrors `TextDoc.materializeAt`'s own
    // `self.holes.items.len != 0` guard, scoped to just this object
    // (unlike `compact`'s whole-doc guard below: a read of ONE object's
    // history has no reason to care about a DIFFERENT object's holes).
    if (self.text_holes.get(obj)) |h| {
        if (h.items.len != 0) return error.Unrealized;
    }

    var w = Walker.initWithBases(&self.history, self.strings.items, &self.text_bases);
    defer w.deinit(gpa);
    var sink: std.ArrayList(Effect) = .empty;
    defer sink.deinit(gpa);
    // Silent (first_new == n: nothing emits) and filtered to `include` —
    // reconstructs `obj`'s state as of `version_token` without walking
    // anything strictly after it. Same discipline as
    // `materializeTextBasesAt`, scoped to one object via `w.seqs`.
    w.replayAll(gpa, n, &sink, include) catch |e| switch (e) {
        error.Corrupt => unreachable, // trusted local history
        else => |err| return err,
    };
    assert(sink.items.len == 0);

    const obj_id = self.history.idOf(obj_lv);
    var sw: SeqWalker = if (w.seqs.fetchRemove(obj_lv)) |kv| kv.value else blk: {
        // `obj` exists as of this version but nothing in `include` ever
        // touched its sequence (e.g. an untouched compacted base, or a
        // just-created empty text object): same fallback
        // `silentObjectReplay` uses for the analogous current-version case.
        var s: SeqWalker = .empty;
        if (self.text_bases.get(obj_id)) |base| {
            try s.initBase(gpa, base.scalars, seq_walker.base_placeholder_lv);
        }
        break :blk s;
    };
    defer sw.deinit(gpa);

    var base_chars: ?[]u21 = null;
    defer if (base_chars) |bc| gpa.free(bc);
    if (self.text_bases.get(obj_id)) |base| {
        base_chars = try decodeUtf8Scalars(gpa, base.bytes);
    }

    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var buf: [4]u8 = undefined;
    var it = sw.s.aliveIterator();
    while (it.next()) |alive| {
        const ch = if (alive.lv == seq_walker.base_placeholder_lv)
            base_chars.?[alive.arena]
        else
            self.history.opOf(alive.lv).text_ins.ch;
        const len = std.unicode.utf8Encode(ch, &buf) catch unreachable;
        try bytes.appendSlice(gpa, buf[0..len]);
    }
    return Rope.fromSlice(gpa, bytes.items);
}

// ── Compaction (text-object compaction — the W7 shape) ───────────────
// delta 2, `stemma-unification.md` §3 step 4. WHOLE-DOC, ONE linearization
// point — the exact TextDoc.compact shape (`heads.len == 1`, everything
// causally before the stable point becomes a frozen base), NOT per-object
// compaction (explicitly out of scope, §4.1 risk 3: compacting anything
// forces a linearization point across the whole tree, on purpose — a
// finer per-object design is a materially different, harder problem this
// step does not attempt).
//
// STRUCTURAL OPS (F3, delta 6, added after this section was written): a
// `struct_create`/`struct_move` anywhere in the causal past of the stable
// point refuses compaction outright too (`error.NotCompactable`, same
// refusal as list structure — see the check below and its doc comment) —
// delta 2b's scope grows to cover them alongside list content.
//
// SCOPE, NAMED PRECISELY: this compacts TEXT-OBJECT content only — the
// shape W7 (weft's `Document` re-basing onto the unified core) actually
// needs, since weft's text buffers are exactly "one `EventGraph` with one
// `SeqWalker`," the degenerate case this generalizes from TextDoc. It
// does NOT compact list content or map-register history. A separate
// follow-up — call it delta 2b — would need an ordered-list-of-values
// base (including nested-object references, in FugueMax document order)
// with no `SeqWalker.initBase` analog today; not attempted here. Plainly:
// a transcript, or any document, with list content in the causal past of
// its chosen stable point CANNOT compact past that point (below); it can
// still compact an EARLIER, list-free point.
//
// ONLY `text_ins`/`text_del` actually compact. Two INDEPENDENT reasons,
// both discovered while implementing (neither visible from the study),
// rule out folding away `map_set`/`map_del`/`list_ins`/`list_del` too:
//
// 1. **`causal.compactGraph`'s per-agent watermark (`seq_base`) can only
//    represent "this agent's compacted events are their EARLIEST N"** —
//    whatever `in_base` a caller supplies MUST be a causal PREFIX of
//    every single agent's own timeline, not an arbitrary per-event
//    subset (`compactGraph`'s `new_graph.add` asserts on it otherwise).
//    TextDoc's `in_base` (the whole causal past of `s`) is automatically
//    such a prefix; the moment `in_base` here excludes SOME op kinds but
//    not others, it is NOT automatically a prefix anymore — an agent
//    could interleave an excluded (map/list) and an included (text)
//    event in either order (creating a text object, then editing it, is
//    the single most common such interleaving). `compact` below
//    EXPLICITLY VALIDATES the prefix property per agent after computing
//    `in_base`, rejecting (`error.NotCompactable`) rather than silently
//    violating it — this is what makes "just exclude map/list, compact
//    text" honest rather than a ticking assertion failure. This check is
//    PER AGENT and structural; it does NOT by itself rule out the
//    cross-agent hazard in reason 2's third bullet below — that needs
//    its own, separate guard.
// 2. **Even where the prefix holds, folding away a `map_set`/`map_del`
//    (or `list_ins`/`list_del`) would break three things a text-only base
//    doesn't have to solve:**
//    - Object identity: `objects_state.Walker` (the replay/apply driver
//      every local edit and merge goes through) dispatches every op by
//      resolving its object reference to a CURRENT `Lv` via
//      `self.history.lvOf` and keys its own per-object state
//      (`Walker.seqs`/`Walker.maps`) by that same `Lv` — there is no
//      compacted-object registry it consults instead. Folding away a
//      `map_set`/`list_ins` that CREATED an object still being edited
//      would make it permanently unaddressable for any FUTURE local edit
//      or merge (`ObjectDoc.zig`'s own `obj_index`, EventId-keyed,
//      already solves this for READS — `objId`/`ref`/`mapGet` — but
//      `objects_state.Walker`'s internals are a separate, harder piece
//      not touched here).
//    - Bootstrap: a fresh replica reconstructing the document PURELY
//      from `serialize`d bytes has no way to learn a compacted-away
//      SCALAR map value either — unlike text (`jw.TextBase`/
//      `SeqWalker.initBase`), nothing here materializes "the current
//      live value of every map key" into a base a bootstrap can read.
//      (Confirmed the hard way: an earlier version of this function DID
//      fold away non-object-creating `map_set`s, and "serialize/open
//      round-trip" silently lost every scalar field set before the
//      compaction point — exactly the failure mode this doc comment
//      warns future changes away from.)
//    - MV-register conflict resolution needs every RETAINED write's true
//      causal relationship to every OTHER retained write to survive —
//      see the cross-agent guard below, and its own doc comment, for the
//      concrete failure this closes (found by review, not designed for
//      up front: compacting text content between two retained map writes
//      can silently sever the causal edge PROVING one supersedes the
//      other, making them look concurrent instead).
//    Building either of the first two — a stable non-`Lv` object key for
//    `objects_state.Walker`, or a map-content base — is real, tractable
//    follow-up work, deliberately not attempted here.
//
// List STRUCTURE (`list_ins`/`list_del`) is ALSO refused outright if any
// exists in the causal past of `s` (`error.NotCompactable`) — even though
// it's excluded from `in_base` just like map ops, a list's FugueMax
// ordering has no base-materialization story the way `SeqWalker.initBase`
// gives text for free (building one — correctly representing an
// arbitrary-length list of values including nested-object references —
// is real, tractable follow-up work; delta 2b, above). A document with
// list content simply can't compact PAST the point that content was last
// touched; compacting an EARLIER, list-free stable point still works
// (`TextDoc.compact`'s "mid-history" shape, mirrored in
// `object_tests.zig`).
//
// Map register conflict resolution (`mapGet`/`setOrder`/
// `mapConflictCount`) stays sound across compaction because of TWO things
// together, not one: `map_set`/`map_del` never compact (every write stays
// a real, resolvable event — §4.1 risk 5's hazard, about FOLDED map
// history, therefore never arises here, because there is no map history
// compaction to trigger it), AND `compact` refuses (below) whenever a
// RETAINED event's causal edge into the base would have to be dropped for
// anything other than the stable point itself — the guard that closes the
// cross-agent hazard just named. Map history staying unfolded is
// necessary but NOT sufficient on its own; the guard is what makes it
// actually sound (see its doc comment for the proof-by-counterexample
// review found, and `object_tests.zig`'s regression coverage).
//
// Text objects compact exactly like TextDoc's one sequence, once per
// object, sharing `causal.compactGraph`'s graph-rebuild and
// `SeqWalker.initBase`/`base_placeholder_lv`'s base-placeholder
// mechanism — the generalization this step is chartered to prove out.

/// One concrete instance of the per-agent-prefix guard above, worth
/// naming plainly: a SINGLE agent that both creates a text object AND
/// later edits it, with that edit in the causal past of `stable_token`,
/// is `error.NotCompactable` (creating `map_set` excluded from `in_base`,
/// editing `text_ins` included, on the same agent's timeline → not a
/// prefix). `object_tests.zig`'s compaction battery always splits
/// "founder creates" from "a different agent edits" for exactly this
/// reason — not stylistic, load-bearing.
pub fn compact(self: *ObjectDoc, gpa: Allocator, stable_token: []const u8) CompactError!void {
    var heads: std.ArrayList(Lv) = .empty;
    defer heads.deinit(gpa);
    try self.decodeVersion(gpa, stable_token, true, &heads);
    if (heads.items.len != 1) return error.NotCompactable;
    const s = heads.items[0];

    const n = self.history.eventCount();
    const past_s = try gpa.alloc(bool, n);
    defer gpa.free(past_s);
    @memset(past_s, false);
    {
        var d = try self.history.diff(gpa, &.{s}, &.{});
        defer d.deinit(gpa);
        for (d.a_only.items) |lv| past_s[lv] = true;
    }

    // Every event OUTSIDE the causal past of `s` must be a genuine
    // descendant of it, never concurrent — else `s` isn't a true
    // linearization point (same check as `TextDoc.compact`).
    for (0..n) |lv| {
        if (past_s[lv]) continue;
        for (self.history.parentsOf(@intCast(lv))) |p| {
            if (past_s[p] and p != s) return error.NotCompactable;
        }
        // List structure inside the causal past of `s` blocks compaction
        // entirely (see the doc comment above) — checked over the WHOLE
        // past (not just this loop's outside-past_s events), so fold it
        // into the same pass rather than a second one.
    }
    for (0..n) |lv| {
        if (!past_s[lv]) continue;
        switch (self.history.opOf(@intCast(lv))) {
            // List structure: see the doc comment above. Structural ops
            // (F3, delta 6) get the SAME safe refusal for the SAME
            // reason — `causal.compactGraph`'s materialization story
            // (`SeqWalker.initBase`) has no analog for the parent-register
            // tree, and folding away a `struct_create`/`struct_move`
            // would break the global-Lamport canonical order the SAME way
            // folding a `map_set` would break `mapConflictCount` (§4.1
            // risk 5's hazard, structural-tree-shaped): a later
            // `merge`/replay needs every retained structural write's true
            // causal relationship to every other one to reproduce cycle
            // rejection identically. A real implementation extending
            // compaction to structural ops is delta 2b's scope, same as
            // list content — not attempted here.
            .list_ins, .list_del, .struct_create, .struct_move => return error.NotCompactable,
            else => {},
        }
    }

    // `in_base`: the causal past of `s`, EXCLUDING every `map_set`/
    // `map_del` event (not just object-creating ones — see the doc
    // comment above `compact` for the two independent reasons: object
    // identity needs a creating event's `Lv` to keep resolving, AND a
    // fresh replica reconstructing the document from `serialize`d bytes
    // ALONE has no other way to learn a compacted-away SCALAR map value
    // either — unlike text, no base-content mechanism captures live map
    // state here; see the doc comment's "what does NOT compact" note).
    // Only `text_ins`/`text_del` compact (list ops already rejected
    // above if present at all in the causal past of `s`).
    const in_base = try gpa.alloc(bool, n);
    defer gpa.free(in_base);
    for (0..n) |lv| {
        in_base[lv] = past_s[lv] and switch (self.history.opOf(@intCast(lv))) {
            .text_ins, .text_del => true,
            else => false,
        };
    }

    // Guard against cross-agent divergence: `causal.compactGraph` drops
    // ANY `in_base` parent edge of a retained event, unconditionally —
    // sound only when every RETAINED event's `in_base` ancestors are
    // exactly {} or {s}. A retained event whose only causal path to
    // another retained event routes THROUGH a folded one would silently
    // lose that edge otherwise. Concretely: retained `map_set` P, folded
    // `text_ins` T (T causally between P and Q), retained `map_set` Q,
    // where Q's supersede-edge back to P routes ONLY through T — each
    // AGENT's own timeline can independently satisfy the prefix property
    // below even though this cross-agent shape is unsound, so this check
    // is separate from (and must run in addition to) that one. Dropping
    // T's edge would make Q's parent list lose the P dependency entirely,
    // so after compaction Q and P look CONCURRENT instead of
    // Q-supersedes-P — `mapConflictCount`/`mapGet`'s deterministic winner
    // both silently diverge from what they were before compacting (and
    // from a fresh replica that reopens the ORIGINAL uncompacted history).
    // Confirmed with an actual probe before this guard existed:
    // `object_tests.zig`'s "compact: refuses when a retained write's only
    // path to another retained write is through a folded text edit"
    // reproduces exactly this. `causal.compactGraph` itself also asserts
    // this invariant now (a second, cheaper line of defense — see its
    // doc comment) but cannot enforce it (it doesn't have the causal-past
    // reachability information to do so); this is the real check.
    for (0..n) |lv| {
        if (in_base[lv]) continue; // only retained events' edges survive
        for (self.history.parentsOf(@intCast(lv))) |p| {
            if (in_base[p] and p != s) return error.NotCompactable;
        }
    }

    // `causal.compactGraph`'s per-agent watermark can only represent
    // "this agent's compacted events are their EARLIEST N" — `in_base`
    // MUST be a causal PREFIX of every single agent's own timeline (see
    // the doc comment above `compact`). This holds automatically when
    // `in_base == past_s` (TextDoc's exact case); excluding creation
    // events on top does NOT hold automatically — an agent could create
    // an object (excluded) and later edit it (included) in either order.
    // Detect and reject rather than silently violate the watermark. This
    // is a DIFFERENT check from the one just above: that one is about
    // whether folding `in_base` loses information ANY retained event
    // needs (cross-agent); this one is about whether `in_base` is even
    // representable by `compactGraph`'s single-watermark-per-agent model
    // at all (single-agent, structural).
    for (self.history.agents.items) |a| {
        var seen_retained = false;
        for (a.lv_by_seq.items) |lv| {
            if (in_base[lv]) {
                if (seen_retained) return error.NotCompactable;
            } else {
                seen_retained = true;
            }
        }
    }

    // Everything below is built FRESH and only swapped into `self` once
    // every fallible step has succeeded (the "Commit" block at the
    // bottom, which does no allocation) — `self` itself stays untouched
    // until then, matching `TextDoc.compact`'s discipline.

    // Materialize every touched text object's alive-as-of-`s` scalar
    // content — same ordering discipline as `TextDoc.compact`
    // (`materializeAt` before the graph rebuild). Keyed by the object's
    // portable `EventId`, so no re-keying is needed after `compactGraph`
    // renumbers `Lv`s below (unlike `Lv`-keyed state — see `node_of`).
    var new_bases = try self.materializeTextBasesAt(gpa, past_s);
    errdefer freeTextBases(gpa, &new_bases);

    var result = try causal.compactGraph(gpa, &self.history, in_base, s);
    errdefer result.graph.deinit(gpa);
    defer gpa.free(result.lv_map);

    const new_base_version = try gpa.dupe(u8, stable_token);
    errdefer gpa.free(new_base_version);

    // Merge the new bases into a fresh table — no re-keying needed
    // (`EventId`-keyed, see above). An object re-compacted a second time
    // replaces its prior (smaller) base; one untouched this round keeps
    // its prior base unchanged. Every entry is `fetchRemove`d out of
    // whichever of `self.text_bases`/`new_bases` it came from as it's
    // transferred, so at any point (including mid-failure) each base's
    // bytes are owned by exactly one of {`self.text_bases`, `new_bases`,
    // `merged_bases`} — never aliased across two of them.
    var merged_bases: jw.TextBaseMap = .empty;
    errdefer freeTextBases(gpa, &merged_bases);
    {
        var old_keys: std.ArrayList(EventId) = .empty;
        defer old_keys.deinit(gpa);
        var kit = self.text_bases.keyIterator();
        while (kit.next()) |k| try old_keys.append(gpa, k.*);
        for (old_keys.items) |key| {
            if (new_bases.fetchRemove(key)) |kv| {
                gpa.free(self.text_bases.fetchRemove(key).?.value.bytes); // superseded
                errdefer gpa.free(kv.value.bytes); // `put` below can still fail (OOM)
                try merged_bases.put(gpa, key, kv.value);
            } else {
                const kv = self.text_bases.fetchRemove(key).?;
                errdefer gpa.free(kv.value.bytes);
                try merged_bases.put(gpa, key, kv.value);
            }
        }
        var new_keys: std.ArrayList(EventId) = .empty;
        defer new_keys.deinit(gpa);
        var nkit = new_bases.keyIterator();
        while (nkit.next()) |k| try new_keys.append(gpa, k.*);
        for (new_keys.items) |key| {
            const kv = new_bases.fetchRemove(key).?;
            errdefer gpa.free(kv.value.bytes);
            try merged_bases.put(gpa, key, kv.value);
        }
    }
    new_bases.deinit(gpa); // drained above; nothing left to free
    new_bases = .empty; // the still-armed `errdefer` above must see a valid,
    // empty (no-op-to-free) map if anything below still fails — it isn't
    // cancelled just because we've already freed this manually here.

    // `node_of` is the one remaining `Lv`-keyed piece of state (a
    // transient, replay-scoped index — see its doc comment); rebuild it
    // against the new `Lv` space. `self.nodes`/`self.obj_index` need no
    // rewriting at all: every identity they hold is already `EventId`,
    // stable across this renumbering by construction (see `Node`'s doc
    // comment).
    var new_node_of: std.ArrayList(u32) = .empty;
    errdefer new_node_of.deinit(gpa);
    try new_node_of.appendNTimes(gpa, no_node, result.graph.eventCount());
    for (self.node_of.items, 0..) |idx, old_lv| {
        if (idx == no_node) continue;
        new_node_of.items[result.lv_map[old_lv]] = idx;
    }

    // Commit — infallible from here on.
    const head_id = self.history.idOf(s);
    self.node_of.deinit(gpa);
    self.node_of = new_node_of;
    self.history.deinit(gpa);
    self.history = result.graph;
    gpa.free(self.base_version);
    self.base_version = new_base_version;
    self.base_head = head_id;
    self.text_bases.deinit(gpa); // drained above; nothing left to free
    self.text_bases = merged_bases;
}

fn freeTextBases(gpa: Allocator, bases: *jw.TextBaseMap) void {
    var it = bases.valueIterator();
    while (it.next()) |b| gpa.free(b.bytes);
    bases.deinit(gpa);
}

/// Adopt ONE bootstrap batch text-base entry's METADATA (bytes + scalar
/// count) into `self.text_bases`. Called from `merge`, for every entry,
/// BEFORE `historyPhase` runs at all — this is the piece
/// `objects_state.Walker.getSeqWalker` reads (`self.text_bases.get(obj)`)
/// the FIRST time replay touches an object, to seed its FugueMax
/// sequence's placeholder items; every `text_ins`/`text_del` EFFECT
/// `historyPhase`'s replay computes for that object is a position
/// relative to those placeholders. Seed this too late (e.g. only once
/// the object's node exists, at its creation effect, which is what an
/// earlier version of this function did) and replay sees ZERO base
/// scalars for the object — for a trivial (empty) base this is
/// harmless, but for a real, non-empty base it corrupts every position
/// computed against it, surfacing as `error.Corrupt` inside
/// `Sequence.applyInsert`/`applyDelete`, not merely a content clobber.
/// Does NOT touch `self.nodes`/any rope — the node this entry's `id`
/// names does not exist yet at the point `merge` calls this (nothing
/// has been added to the graph, let alone the tree, until AFTER this
/// runs); materializing the rope is `adoptTextBaseRope`'s job, run later
/// once the node exists. Keyed directly by the wire entry's portable
/// identity — no `lvOf` resolution needed at all (see `jw.TextBaseMap`'s
/// doc comment).
fn adoptTextBaseMetadata(self: *ObjectDoc, gpa: Allocator, id: EventId, tb: Decoder.TextBaseRef) Allocator.Error!void {
    // Build the fully-owned replacement BEFORE touching `self.text_bases`
    // at all: `getOrPut` inserts the KEY unconditionally, before any
    // value is known, so if a value is computed only AFTER that, a later
    // allocation failure leaves a hashmap entry pointing at uninitialized
    // memory — `deinit`'s free-every-value loop then crashes on it.
    // Fallible work first, commit last.
    const bytes_owned = try gpa.dupe(u8, tb.bytes);
    errdefer gpa.free(bytes_owned);
    const gop = try self.text_bases.getOrPut(gpa, id);
    if (gop.found_existing) gpa.free(gop.value_ptr.bytes); // defensive; unreachable on a true bootstrap
    gop.value_ptr.* = .{ .bytes = bytes_owned, .scalars = tb.scalars };
}

/// Materialize `id`'s already-adopted base bytes (`adoptTextBaseMetadata`
/// must have already run for `id` — always true here: `merge` seeds
/// METADATA for every batch base entry before `historyPhase` runs at
/// all, see its own doc comment) into its just-created node's rope.
///
/// `ADDPEER_BOOTSTRAP_HAZARD` (weft's own name for this, W7a report):
/// this must run for `id` BEFORE any `.text_ins`/`.text_del` effect
/// targeting `id` is applied to its rope — a single bootstrap batch can
/// legitimately carry both a base AND live edits authored past it (e.g.
/// `serialize()` of a doc whose founder base is trivial but has since
/// been typed into), and those edits' rope offsets are computed against
/// a rope that STARTS FROM the seeded base, not an empty one. Calling
/// this from `merge`'s tree-application loop right after the object's
/// creation effect (`.map_add`/`.list_ins` with `.new_text`) — rather
/// than once, deferred, after the whole effects list has already run —
/// is what guarantees that ordering: creation and its rope-seed are
/// adjacent, so every subsequent effect for `id` in the (causally
/// ordered) effects list necessarily lands on the seeded rope, never the
/// reverse.
fn adoptTextBaseRope(self: *ObjectDoc, gpa: Allocator, id: EventId) Allocator.Error!void {
    const tb = self.text_bases.get(id).?;
    var rope = try Rope.fromSlice(gpa, tb.bytes);
    errdefer rope.deinit(gpa);
    const node_idx = self.obj_index.get(id).?;
    self.nodes.items[node_idx].text.rope.deinit(gpa);
    self.nodes.items[node_idx].text.rope = rope;
}

/// Materialize the alive-as-of-`past_s` scalar content of every text
/// object touched within `past_s`, keyed by their PORTABLE creation
/// identity. Seeds from any EXISTING `self.text_bases` entries first
/// (progressive re-compaction: a second `compact()` call extends, rather
/// than replaces from scratch, an already-compacted object's base).
/// `error.Unrealized` if ANY object anywhere in the document still has an
/// active hole (`compact`'s own whole-doc-linearization-point granularity
/// — coarser than `materializeAt`'s per-object guard, but `compact`
/// itself only ever operates whole-doc, so there is no finer question to
/// ask here): decoding `old_decoded`'s base-placeholder scalars below
/// would otherwise read a hole's zero-filled placeholder bytes as if they
/// were real pristine content — silently WRONG (still valid UTF-8: NUL
/// codepoints), not merely unavailable, which is exactly what this guard
/// is for. This is `compact`'s "transitive" `Unrealized` refusal — the
/// one call site `compact` routes its materialization through.
fn materializeTextBasesAt(self: *const ObjectDoc, gpa: Allocator, past_s: []const bool) (Allocator.Error || error{Unrealized})!jw.TextBaseMap {
    if (self.text_holes.count() != 0) return error.Unrealized;
    // Decode every EXISTING base's bytes to scalars once, up front — a
    // still-alive base placeholder's arena index is exactly its position
    // in that decoded array (see `Sequence.initBase`: placeholders are
    // appended in order, arena 0..count-1, and arena identity never
    // changes even if some are later toggled dead).
    var old_decoded: std.AutoHashMapUnmanaged(EventId, []u21) = .empty;
    defer {
        var it = old_decoded.valueIterator();
        while (it.next()) |d| gpa.free(d.*);
        old_decoded.deinit(gpa);
    }
    {
        var it = self.text_bases.iterator();
        while (it.next()) |e| {
            try old_decoded.put(gpa, e.key_ptr.*, try decodeUtf8Scalars(gpa, e.value_ptr.bytes));
        }
    }

    var w = Walker.initWithBases(&self.history, self.strings.items, &self.text_bases);
    defer w.deinit(gpa);
    var sink: std.ArrayList(Effect) = .empty;
    defer sink.deinit(gpa);
    const n: Lv = @intCast(self.history.eventCount());
    // Silent (first_new == n: nothing emits) and filtered to `past_s` —
    // this reconstructs per-object `SeqWalker` state as of `s` without
    // walking anything strictly after it.
    w.replayAll(gpa, n, &sink, past_s) catch |e| switch (e) {
        error.Corrupt => unreachable, // trusted local history
        else => |err| return err,
    };
    assert(sink.items.len == 0);

    var out: jw.TextBaseMap = .empty;
    errdefer freeTextBases(gpa, &out);
    var seq_it = w.seqs.iterator();
    while (seq_it.next()) |entry| {
        const obj_lv = entry.key_ptr.*;
        if (self.objKind(obj_lv) != .text) continue; // lists: never based
        const obj_id = self.history.idOf(obj_lv);
        const sw = entry.value_ptr;
        var bytes: std.ArrayList(u8) = .empty;
        errdefer bytes.deinit(gpa);
        var scalars: usize = 0;
        var buf: [4]u8 = undefined;
        var it = sw.s.aliveIterator();
        while (it.next()) |alive| {
            const ch = if (alive.lv == seq_walker.base_placeholder_lv)
                old_decoded.get(obj_id).?[alive.arena]
            else
                self.history.opOf(alive.lv).text_ins.ch;
            const len = std.unicode.utf8Encode(ch, &buf) catch unreachable;
            try bytes.appendSlice(gpa, buf[0..len]);
            scalars += 1;
        }
        const owned = try bytes.toOwnedSlice(gpa);
        errdefer gpa.free(owned); // `out.put` below can still fail (OOM)
        try out.put(gpa, obj_id, .{ .bytes = owned, .scalars = scalars });
    }
    return out;
}

fn decodeUtf8Scalars(gpa: Allocator, bytes: []const u8) Allocator.Error![]u21 {
    var out: std.ArrayList(u21) = .empty;
    errdefer out.deinit(gpa);
    var it = (std.unicode.Utf8View.init(bytes) catch unreachable).iterator();
    while (it.nextCodepoint()) |ch| try out.append(gpa, ch);
    return out.toOwnedSlice(gpa);
}

const ObjKind = enum { map, list, text };

fn objKind(self: *const ObjectDoc, creation_lv: Lv) ObjKind {
    const val: ValPayload = switch (self.history.opOf(creation_lv)) {
        .map_set => |m| m.val,
        .list_ins => |l| l.val,
        else => unreachable,
    };
    return switch (val) {
        .new_map => .map,
        .new_list => .list,
        .new_text => .text,
        else => unreachable,
    };
}

fn historyPhase(
    self: *ObjectDoc,
    gpa: Allocator,
    dec: *const Decoder,
    aids: []const AgentId,
    effects: *std.ArrayList(Effect),
) MergeError!bool {
    const pre_events = self.history.events.items.len;
    const pre_pool = self.history.parents_pool.items.len;
    const pre_strings = self.strings.items.len;
    const pre_frontier = try gpa.dupe(Lv, self.history.frontier.items);
    defer gpa.free(pre_frontier);
    const pre_seq_lens = try gpa.alloc(usize, self.history.agents.items.len);
    defer gpa.free(pre_seq_lens);
    for (self.history.agents.items, pre_seq_lens) |a, *len| len.* = a.lv_by_seq.items.len;
    errdefer {
        self.history.events.items.len = pre_events;
        self.history.parents_pool.items.len = pre_pool;
        self.strings.items.len = pre_strings;
        for (self.history.agents.items, 0..) |*a, i| {
            a.lv_by_seq.items.len = if (i < pre_seq_lens.len) pre_seq_lens[i] else 0;
        }
        self.history.frontier.clearRetainingCapacity();
        self.history.frontier.appendSliceAssumeCapacity(pre_frontier);
    }

    const first_new: Lv = @intCast(self.history.eventCount());
    var any_new = false;
    for (dec.events.items) |ev| {
        const id: EventId = .{ .agent = aids[ev.agent_idx], .seq = ev.seq };
        if (self.history.isKnown(id)) continue;
        var parent_lvs: std.ArrayList(Lv) = .empty;
        defer parent_lvs.deinit(gpa);
        for (dec.parentsOf(ev)) |pref| {
            const pid: EventId = .{ .agent = aids[pref.agent_idx], .seq = pref.seq };
            if (self.history.lvOf(pid)) |plv| {
                try parent_lvs.append(gpa, plv);
            }
            // else: validated (`Decoder.validate`) to be this doc's base
            // head — implicit, same discipline as `TextDoc.historyPhase`.
        }
        const op = try self.internOp(gpa, ev, aids);
        _ = try self.history.add(gpa, id, parent_lvs.items, op);
        any_new = true;
    }
    if (!any_new) return false;

    var w = Walker.initWithBases(&self.history, self.strings.items, &self.text_bases);
    defer w.deinit(gpa);
    try w.replayAll(gpa, first_new, effects, null);
    // Rejects (whole-batch) any transformed text effect landing interior
    // to an unrealized base span — rides the errdefer rollback armed at
    // the top of this function (the tree-application loop `merge` drives
    // over `effects` runs strictly later, in `merge` itself, after this
    // function returns — nothing has touched `self.nodes` yet at this
    // point, so there is nothing beyond the graph mutation above for the
    // errdefer to undo).
    try self.checkHoleConflicts(gpa, effects.items);
    return true;
}

/// Reject (whole-batch) any transformed `.text_ins`/`.text_del` effect
/// that lands interior to an unrealized base span of the object it
/// targets — the per-object analog of `TextDoc.checkHoleConflicts`,
/// scoped to whichever objects this batch's effects touch AND have
/// active holes (`self.text_holes` misses for every other object — free,
/// same as every other per-object hole lookup here).
///
/// COORDINATE SPACE (the trap): `effects` entries carry positions in the
/// state produced by ALL PREVIOUSLY EMITTED effects (`Effect`'s own doc
/// comment) — i.e. each object's "current" scalar space drifts as
/// EARLIER effects on that SAME object land. A hole's global scalar
/// start must be tracked incrementally through the stream exactly like
/// `TextDoc.checkHoleConflicts` tracks it for the whole document: an
/// insert at-or-before a hole's start shifts it right by one; a delete
/// strictly before it shifts it left by one. Ported here per object
/// (`starts`, one entry per object that HAS active holes) instead of
/// once for the whole document, since two different objects' holes are
/// never affected by each other's effects.
fn checkHoleConflicts(self: *const ObjectDoc, gpa: Allocator, effects: []const Effect) MergeError!void {
    if (self.text_holes.count() == 0) return;

    var starts: std.AutoHashMapUnmanaged(EventId, []u64) = .empty;
    defer {
        var it = starts.valueIterator();
        while (it.next()) |s| gpa.free(s.*);
        starts.deinit(gpa);
    }
    var hit = self.text_holes.iterator();
    while (hit.next()) |entry| {
        const holes = entry.value_ptr.items;
        if (holes.len == 0) continue;
        const node_idx = self.obj_index.get(entry.key_ptr.*).?;
        const rope = &self.nodes.items[node_idx].text.rope;
        const arr = try gpa.alloc(u64, holes.len);
        var acc: u64 = 0;
        for (holes, arr) |h, *s| {
            s.* = @as(u64, rope.offsetToScalar(h.cur_offset)) + acc;
            acc += h.scalars;
        }
        try starts.put(gpa, entry.key_ptr.*, arr);
    }

    for (effects) |eff| switch (eff) {
        .text_ins => |x| {
            const obj_id = self.history.idOf(x.obj);
            const holes = self.text_holes.get(obj_id) orelse continue;
            const s = starts.get(obj_id) orelse continue;
            for (holes.items, s) |h, *sv| {
                if (x.pos > sv.* and x.pos < sv.* + h.scalars) return error.Unrealized;
                if (x.pos <= sv.*) sv.* += 1;
            }
        },
        .text_del => |x| {
            const obj_id = self.history.idOf(x.obj);
            const holes = self.text_holes.get(obj_id) orelse continue;
            const s = starts.get(obj_id) orelse continue;
            for (holes.items, s) |h, *sv| {
                if (x.pos >= sv.* and x.pos < sv.* + h.scalars) return error.Unrealized;
                if (x.pos < sv.*) sv.* -= 1;
            }
        },
        else => {},
    };
}

/// Build the stored op from a decoded one: intern strings, resolve
/// object refs from batch (agent_idx, seq) to EventIds.
fn internOp(self: *ObjectDoc, gpa: Allocator, ev: Decoder.Event, aids: []const AgentId) MergeError!ObjectOp {
    const obj: ?ObjId = if (ev.obj) |o| .{ .agent = aids[o.agent_idx], .seq = o.seq } else null;
    return switch (ev.op_tag) {
        .map_set => .{ .map_set = .{
            .obj = obj,
            .key = try self.intern(gpa, ev.key),
            .val = try self.internVal(gpa, ev.val.?),
        } },
        .map_del => .{ .map_del = .{ .obj = obj, .key = try self.intern(gpa, ev.key) } },
        .list_ins => .{ .list_ins = .{
            .obj = obj.?,
            .pos = ev.pos,
            .val = try self.internVal(gpa, ev.val.?),
        } },
        .list_del => .{ .list_del = .{ .obj = obj.?, .pos = ev.pos } },
        .text_ins => .{ .text_ins = .{ .obj = obj.?, .pos = ev.pos, .ch = ev.ch } },
        .text_del => .{ .text_del = .{ .obj = obj.?, .pos = ev.pos } },
        .struct_create => .{ .struct_create = .{
            .parent = self.structRefFrom(ev.struct_parent, aids),
            .order_key = try self.intern(gpa, ev.order_key),
        } },
        .struct_move => .{ .struct_move = .{
            .node = .{ .agent = aids[ev.struct_node.agent_idx], .seq = ev.struct_node.seq },
            .parent = self.structRefFrom(ev.struct_parent, aids),
            .order_key = try self.intern(gpa, ev.order_key),
        } },
    };
}

fn structRefFrom(self: *const ObjectDoc, wire_ref: Decoder.StructRefWire, aids: []const AgentId) StructRef {
    _ = self;
    return switch (wire_ref) {
        .root => .root,
        .trash => .trash,
        .node => |o| .{ .node = .{ .agent = aids[o.agent_idx], .seq = o.seq } },
    };
}

fn internVal(self: *ObjectDoc, gpa: Allocator, v: Decoder.RawVal) Allocator.Error!ValPayload {
    return switch (v) {
        .null_ => .null_,
        .bool_ => |b| .{ .bool_ = b },
        .int => |x| .{ .int = x },
        .float => |x| .{ .float = x },
        .str => |s| .{ .str = try self.intern(gpa, s) },
        .new_map => .new_map,
        .new_list => .new_list,
        .new_text => .new_text,
    };
}

// ── Wire format ─────────────────────────────────────────────────────
// v1 "stj" 0x01: uv agent_count, per agent (uv name_len, name);
// uv event_count, per event: uv agent_idx, uv seq, uv parent_count,
// parents (uv agent_idx, uv seq); u8 op tag; obj ref (u8 0 = root,
// 1 + uv agent_idx + uv seq); op-specific fields. Ints are zigzag'd.
// v2 "stj" 0x02 (emitted whenever `base_version.len > 0` — via `compact`
// OR via `openFromContent`'s bulk load, see `object_magic_v2`):
//   uv agent_count, per agent: uv name_len, name, uv seq_base   (v1 +
//     seq_base per agent; 0 for an agent with nothing compacted)
//   uv base_version_len, base_version bytes                     (a
//     `version()`-shaped token — self-contained, no table needed)
//   uv text_base_count, per entry: uv agent_idx, uv seq,        (the
//     text object's creation event — always present as a real event in
//     THIS SAME batch or already known to the receiver, see
//     `Decoder.validate`'s bootstrap check)
//     uv scalars, uv bytes_len, bytes                           (its
//     compacted pre-history, UTF-8 — `jw.TextBase`)
//   uv event_count, per event: same as v1 (map/list events are NEVER
//     compacted — see `compact`'s doc comment — so nothing about the
//     event stream itself changes shape between v1 and v2)
// v3 "stj" 0x03 (emitted instead of v2 whenever the sender has any active
// hole — partial checkout, `openPartial`): identical to v2 EXCEPT
// `uv text_base_count` is always encoded as 0 and no entries follow —
// "version-only": `base_version` is still present (same-base peers merge
// normally) but this batch can never seed a bootstrap (`Decoder`'s
// `BaseSection.full` decodes false only for v3). See the wire-contract
// doc comment above `openPartial` for the full rationale.

// F3, delta 6 (`stemma-unification.md` §3 step 5): `struct_create`/
// `struct_move` are ADDITIVE new tags (6/7) — a doc with no structural ops
// never emits them, so its wire bytes are byte-identical to before this
// step; a decoder built before this step would reject bytes that DO carry
// them (an unrecognized tag), which is the expected, one-directional
// shape of "additive" (old bytes still decode under the NEW decoder; new
// bytes are not required to decode under an OLD one). Payload, appended
// after the generic `has_obj` byte (always 0 for these two tags — see
// `opObj`):
//   struct_create: struct-ref(parent), uv order_key_len, order_key bytes
//   struct_move:   uv node_agent_idx, uv node_seq, struct-ref(parent),
//                  uv order_key_len, order_key bytes
// struct-ref: u8 tag (0=root, 1=trash, 2=node), node only: uv agent_idx,
//   uv seq.
const OpTag = enum(u8) { map_set = 0, map_del = 1, list_ins = 2, list_del = 3, text_ins = 4, text_del = 5, struct_create = 6, struct_move = 7 };
const ValTag = enum(u8) { null_ = 0, false_ = 1, true_ = 2, int = 3, float = 4, str = 5, new_map = 6, new_list = 7, new_text = 8 };

fn encodeEvents(self: *const ObjectDoc, gpa: Allocator, lvs: []const Lv) Allocator.Error![]u8 {
    const compacted = self.base_version.len > 0;
    // Partial checkout (`openPartial`/`text_holes`): a holey sender emits
    // v3 instead of v2 — `base_version` only, ZERO `text_base` entries,
    // NEVER bootstrap-capable (`Decoder.BaseSection.full` decodes false)
    // — see the "Partial checkout" section's wire-contract doc comment
    // above `openPartial` for the full rationale.
    const holey = self.text_holes.count() != 0;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, if (!compacted) object_magic_v1 else if (holey) object_magic_v3 else object_magic_v2);

    var table: std.ArrayList(AgentId) = .empty;
    defer table.deinit(gpa);
    for (lvs) |lv| {
        try tableAdd(gpa, &table, self.history.idOf(lv).agent);
        for (self.history.parentsOf(lv)) |p| try tableAdd(gpa, &table, self.history.idOf(p).agent);
        const op = self.history.opOf(lv);
        if (opObj(op)) |o| try tableAdd(gpa, &table, o.agent);
        switch (op) {
            .struct_create => |c| if (structRefAgent(c.parent)) |a| try tableAdd(gpa, &table, a),
            .struct_move => |m| {
                try tableAdd(gpa, &table, m.node.agent);
                if (structRefAgent(m.parent)) |a| try tableAdd(gpa, &table, a);
            },
            else => {},
        }
    }
    if (compacted and !holey) {
        var bit = self.text_bases.keyIterator();
        while (bit.next()) |k| try tableAdd(gpa, &table, k.agent);
    }
    if (compacted) {
        // A bootstrapping receiver needs to resolve `base_head`'s agent
        // (`versionSingleEntry(base_version).name`) BEFORE any event or
        // text-base entry has registered it for them — e.g. when `s` is
        // authored by an agent whose every event got folded into another
        // object's base (nothing else in the batch would ever mention
        // them). Mirrors `TextDoc.encodeEvents`'s identical `base_head`
        // table entry. Still needed for v3 (a same-base peer's parent
        // resolution may reference it).
        if (self.base_head) |h| try tableAdd(gpa, &table, h.agent);
    }
    try putUv(gpa, &out, table.items.len);
    for (table.items) |aid| {
        const name = self.history.agentName(aid);
        try putUv(gpa, &out, name.len);
        try out.appendSlice(gpa, name);
        if (compacted) try putUv(gpa, &out, self.history.agents.items[@intFromEnum(aid)].seq_base);
    }

    if (compacted) {
        try putUv(gpa, &out, self.base_version.len);
        try out.appendSlice(gpa, self.base_version);
        if (holey) {
            try putUv(gpa, &out, 0); // version-only: zero text_base entries, ever
        } else {
            try putUv(gpa, &out, self.text_bases.count());
            var it = self.text_bases.iterator();
            while (it.next()) |e| {
                const id = e.key_ptr.*;
                try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
                try putUv(gpa, &out, id.seq);
                try putUv(gpa, &out, e.value_ptr.scalars);
                try putUv(gpa, &out, e.value_ptr.bytes.len);
                try out.appendSlice(gpa, e.value_ptr.bytes);
            }
        }
    }

    try putUv(gpa, &out, lvs.len);
    for (lvs) |lv| {
        const id = self.history.idOf(lv);
        try putUv(gpa, &out, tableIndexOf(table.items, id.agent));
        try putUv(gpa, &out, id.seq);
        const parents = self.history.parentsOf(lv);
        try putUv(gpa, &out, parents.len);
        for (parents) |p| {
            const pid = self.history.idOf(p);
            try putUv(gpa, &out, tableIndexOf(table.items, pid.agent));
            try putUv(gpa, &out, pid.seq);
        }
        const op = self.history.opOf(lv);
        try out.append(gpa, @intFromEnum(@as(OpTag, switch (op) {
            .map_set => .map_set,
            .map_del => .map_del,
            .list_ins => .list_ins,
            .list_del => .list_del,
            .text_ins => .text_ins,
            .text_del => .text_del,
            .struct_create => .struct_create,
            .struct_move => .struct_move,
        })));
        if (opObj(op)) |o| {
            try out.append(gpa, 1);
            try putUv(gpa, &out, tableIndexOf(table.items, o.agent));
            try putUv(gpa, &out, o.seq);
        } else {
            try out.append(gpa, 0);
        }
        switch (op) {
            .map_set => |m| {
                try putUv(gpa, &out, m.key.len);
                try out.appendSlice(gpa, self.str(m.key));
                try self.encodeVal(gpa, &out, m.val);
            },
            .map_del => |m| {
                try putUv(gpa, &out, m.key.len);
                try out.appendSlice(gpa, self.str(m.key));
            },
            .list_ins => |l| {
                try putUv(gpa, &out, l.pos);
                try self.encodeVal(gpa, &out, l.val);
            },
            .list_del => |l| try putUv(gpa, &out, l.pos),
            .text_ins => |x| {
                try putUv(gpa, &out, x.pos);
                try putUv(gpa, &out, x.ch);
            },
            .text_del => |x| try putUv(gpa, &out, x.pos),
            .struct_create => |c| {
                try self.encodeStructRef(gpa, &out, table.items, c.parent);
                try putUv(gpa, &out, c.order_key.len);
                try out.appendSlice(gpa, self.str(c.order_key));
            },
            .struct_move => |m| {
                try putUv(gpa, &out, tableIndexOf(table.items, m.node.agent));
                try putUv(gpa, &out, m.node.seq);
                try self.encodeStructRef(gpa, &out, table.items, m.parent);
                try putUv(gpa, &out, m.order_key.len);
                try out.appendSlice(gpa, self.str(m.order_key));
            },
        }
    }
    return out.toOwnedSlice(gpa);
}

fn opObj(op: ObjectOp) ?ObjId {
    return switch (op) {
        .map_set => |m| m.obj,
        .map_del => |m| m.obj,
        .list_ins => |l| l.obj,
        .list_del => |l| l.obj,
        .text_ins => |x| x.obj,
        .text_del => |x| x.obj,
        // Structural ops encode their own refs separately (`struct_move`
        // has TWO — `node` and `parent` — and `struct_create` has none of
        // this generic single-obj shape at all) — see the `OpTag` doc
        // comment.
        .struct_create, .struct_move => null,
    };
}

fn structRefAgent(r: StructRef) ?AgentId {
    return switch (r) {
        .root, .trash => null,
        .node => |id| id.agent,
    };
}

fn encodeStructRef(self: *const ObjectDoc, gpa: Allocator, out: *std.ArrayList(u8), table: []const AgentId, r: StructRef) Allocator.Error!void {
    _ = self;
    switch (r) {
        .root => try out.append(gpa, 0),
        .trash => try out.append(gpa, 1),
        .node => |id| {
            try out.append(gpa, 2);
            try putUv(gpa, out, tableIndexOf(table, id.agent));
            try putUv(gpa, out, id.seq);
        },
    }
}

fn encodeVal(self: *const ObjectDoc, gpa: Allocator, out: *std.ArrayList(u8), v: ValPayload) Allocator.Error!void {
    switch (v) {
        .null_ => try out.append(gpa, @intFromEnum(ValTag.null_)),
        .bool_ => |b| try out.append(gpa, @intFromEnum(if (b) ValTag.true_ else ValTag.false_)),
        .int => |x| {
            try out.append(gpa, @intFromEnum(ValTag.int));
            try putUv(gpa, out, zigzag(x));
        },
        .float => |x| {
            try out.append(gpa, @intFromEnum(ValTag.float));
            try putUv(gpa, out, @bitCast(x));
        },
        .str => |s| {
            try out.append(gpa, @intFromEnum(ValTag.str));
            try putUv(gpa, out, s.len);
            try out.appendSlice(gpa, self.str(s));
        },
        .new_map => try out.append(gpa, @intFromEnum(ValTag.new_map)),
        .new_list => try out.append(gpa, @intFromEnum(ValTag.new_list)),
        .new_text => try out.append(gpa, @intFromEnum(ValTag.new_text)),
    }
}

const tableAdd = core.tableAdd;
const tableIndexOf = core.tableIndexOf;

fn zigzag(x: i64) u64 {
    return @bitCast((x << 1) ^ (x >> 63));
}

fn unzigzag(x: u64) i64 {
    const v: i64 = @bitCast(x >> 1);
    return if (x & 1 != 0) ~v else v;
}

const Decoder = struct {
    const ObjRef = struct { agent_idx: u32, seq: u64 };
    const RawVal = union(enum) {
        null_,
        bool_: bool,
        int: i64,
        float: f64,
        str: []const u8, // borrowed from input
        new_map,
        new_list,
        new_text,
    };
    /// Decoded `StructRef` (F3, delta 6) — `node`'s agent index is only
    /// meaningful relative to THIS batch's table, resolved to a real
    /// `AgentId` (via `aids`) in `ObjectDoc.structRefFrom`, same
    /// two-phase discipline as every other batch-table-relative ref.
    const StructRefWire = union(enum) { root, trash, node: ObjRef };
    const Event = struct {
        agent_idx: u32,
        seq: u64,
        parents_start: u32,
        parents_len: u32,
        op_tag: OpTag,
        obj: ?ObjRef,
        key: []const u8 = &.{}, // borrowed
        pos: u64 = 0,
        ch: u21 = 0,
        val: ?RawVal = null,
        // struct_create/struct_move only:
        struct_node: ObjRef = .{ .agent_idx = 0, .seq = 0 },
        struct_parent: StructRefWire = .root,
        order_key: []const u8 = &.{}, // borrowed
    };
    /// One decoded `jw.TextBase` wire entry — `obj` names its text
    /// object's creation event (by batch table index; resolved to an
    /// `Lv` only after `historyPhase` has added it, see
    /// `ObjectDoc.adoptTextBases`). `bytes` borrows from the input.
    const TextBaseRef = struct { obj: ObjRef, scalars: usize, bytes: []const u8 };
    const BaseSection = struct {
        version: []const u8, // borrowed — a `version()`-shaped token
        text_bases: []const TextBaseRef,
        /// False for a v3 ("version-only") base section: `text_bases` is
        /// always empty in that case, but emptiness alone doesn't mean
        /// "nothing was ever compacted here" — `full` is what `merge`'s
        /// bootstrap check actually gates on (see the wire-contract doc
        /// comment above `openPartial`).
        full: bool,
    };

    names: std.ArrayList([]const u8) = .empty,
    /// Per agent, `0` for v1 batches (never compacted) or an uncompacted
    /// agent in a v2 batch.
    seq_bases: std.ArrayList(u64) = .empty,
    base: ?BaseSection = null,
    text_base_pool: std.ArrayList(TextBaseRef) = .empty,
    events: std.ArrayList(Event) = .empty,
    parents_pool: std.ArrayList(ObjRef) = .empty,

    fn parentsOf(self: *const Decoder, ev: Event) []const ObjRef {
        return self.parents_pool.items[ev.parents_start..][0..ev.parents_len];
    }

    fn deinit(self: *Decoder, gpa: Allocator) void {
        self.names.deinit(gpa);
        self.seq_bases.deinit(gpa);
        self.text_base_pool.deinit(gpa);
        self.events.deinit(gpa);
        self.parents_pool.deinit(gpa);
    }

    fn init(gpa: Allocator, bytes: []const u8) ObjectDoc.MergeError!Decoder {
        var self: Decoder = .{};
        errdefer self.deinit(gpa);
        var cur: []const u8 = bytes;
        if (cur.len < object_magic_v1.len or !std.mem.startsWith(u8, cur, "stj")) return error.Corrupt;
        const wire_version = cur[3];
        if (wire_version < 1 or wire_version > 3) return error.Corrupt;
        cur = cur[object_magic_v1.len..];

        const agent_count = try getUv(&cur);
        if (agent_count > 1 << 20) return error.Corrupt;
        for (0..agent_count) |_| {
            const name = try getBytes(&cur, 4096);
            if (name.len == 0) return error.Corrupt;
            try self.names.append(gpa, name);
            try self.seq_bases.append(gpa, if (wire_version >= 2) try getUv(&cur) else 0);
        }

        if (wire_version >= 2) {
            const vlen = try getUv(&cur);
            if (vlen == 0 or vlen > cur.len) return error.Corrupt;
            const vtoken = cur[0..vlen];
            cur = cur[vlen..];
            _ = try versionSingleEntry(vtoken); // must be a single head

            const tb_count = try getUv(&cur);
            if (tb_count > 1 << 20) return error.Corrupt;
            const tb_start: u32 = @intCast(self.text_base_pool.items.len);
            for (0..tb_count) |_| {
                const oaidx = try getUv(&cur);
                if (oaidx >= self.names.items.len) return error.Corrupt;
                const oseq = try getUv(&cur);
                const scalars = try getUv(&cur);
                const blen = try getUv(&cur);
                if (blen > cur.len) return error.Corrupt;
                const tbytes = cur[0..blen];
                cur = cur[blen..];
                if (!std.unicode.utf8ValidateSlice(tbytes)) return error.Corrupt;
                if ((std.unicode.utf8CountCodepoints(tbytes) catch return error.Corrupt) != scalars) {
                    return error.Corrupt;
                }
                try self.text_base_pool.append(gpa, .{
                    .obj = .{ .agent_idx = @intCast(oaidx), .seq = oseq },
                    .scalars = @intCast(scalars),
                    .bytes = tbytes,
                });
            }
            self.base = .{
                .version = vtoken,
                .text_bases = self.text_base_pool.items[tb_start..],
                .full = wire_version == 2,
            };
        }

        const event_count = try getUv(&cur);
        for (0..event_count) |_| {
            var ev: Event = undefined;
            const aidx = try getUv(&cur);
            if (aidx >= self.names.items.len) return error.Corrupt;
            ev.agent_idx = @intCast(aidx);
            ev.seq = try getUv(&cur);
            const pcount = try getUv(&cur);
            if (pcount > 1 << 16) return error.Corrupt;
            ev.parents_start = @intCast(self.parents_pool.items.len);
            ev.parents_len = @intCast(pcount);
            for (0..pcount) |_| {
                const paidx = try getUv(&cur);
                if (paidx >= self.names.items.len) return error.Corrupt;
                try self.parents_pool.append(gpa, .{ .agent_idx = @intCast(paidx), .seq = try getUv(&cur) });
            }
            if (cur.len == 0) return error.Corrupt;
            const tag_byte = cur[0];
            cur = cur[1..];
            if (tag_byte > 7) return error.Corrupt;
            ev.op_tag = @enumFromInt(tag_byte);
            if (cur.len == 0) return error.Corrupt;
            const has_obj = cur[0];
            cur = cur[1..];
            if (has_obj > 1) return error.Corrupt;
            ev.obj = if (has_obj == 1) blk: {
                const oaidx = try getUv(&cur);
                if (oaidx >= self.names.items.len) return error.Corrupt;
                break :blk .{ .agent_idx = @intCast(oaidx), .seq = try getUv(&cur) };
            } else null;
            // Root refs are only legal for map ops and structural ops
            // (struct_create/struct_move encode their own refs separately
            // — see the `OpTag` doc comment; they never use the generic
            // `has_obj` slot at all, so `ev.obj` is always null for them).
            if (ev.obj == null and ev.op_tag != .map_set and ev.op_tag != .map_del and
                ev.op_tag != .struct_create and ev.op_tag != .struct_move) return error.Corrupt;
            ev.key = &.{};
            ev.pos = 0;
            ev.ch = 0;
            ev.val = null;
            switch (ev.op_tag) {
                .map_set => {
                    ev.key = try getBytes(&cur, 4096);
                    if (!std.unicode.utf8ValidateSlice(ev.key)) return error.Corrupt;
                    ev.val = try decodeVal(&cur);
                },
                .map_del => {
                    ev.key = try getBytes(&cur, 4096);
                    if (!std.unicode.utf8ValidateSlice(ev.key)) return error.Corrupt;
                },
                .list_ins => {
                    ev.pos = try getUv(&cur);
                    ev.val = try decodeVal(&cur);
                },
                .list_del => ev.pos = try getUv(&cur),
                .text_ins => {
                    ev.pos = try getUv(&cur);
                    const ch = try getUv(&cur);
                    if (ch > std.math.maxInt(u21) or !std.unicode.utf8ValidCodepoint(@intCast(ch)))
                        return error.Corrupt;
                    ev.ch = @intCast(ch);
                },
                .text_del => ev.pos = try getUv(&cur),
                .struct_create => {
                    ev.struct_parent = try decodeStructRef(&cur, self.names.items.len);
                    ev.order_key = try getBytes(&cur, max_order_key_len);
                },
                .struct_move => {
                    const naidx = try getUv(&cur);
                    if (naidx >= self.names.items.len) return error.Corrupt;
                    const nseq = try getUv(&cur);
                    ev.struct_node = .{ .agent_idx = @intCast(naidx), .seq = nseq };
                    ev.struct_parent = try decodeStructRef(&cur, self.names.items.len);
                    ev.order_key = try getBytes(&cur, max_order_key_len);
                },
            }
            try self.events.append(gpa, ev);
        }
        return self;
    }

    fn decodeVal(cur: *[]const u8) error{Corrupt}!RawVal {
        if (cur.len == 0) return error.Corrupt;
        const tag = cur.*[0];
        cur.* = cur.*[1..];
        if (tag > 8) return error.Corrupt;
        return switch (@as(ValTag, @enumFromInt(tag))) {
            .null_ => .null_,
            .false_ => .{ .bool_ = false },
            .true_ => .{ .bool_ = true },
            .int => .{ .int = unzigzag(try getUv(cur)) },
            .float => .{ .float = @bitCast(try getUv(cur)) },
            .str => blk: {
                const s = try getBytes(cur, 1 << 24);
                if (!std.unicode.utf8ValidateSlice(s)) return error.Corrupt;
                break :blk .{ .str = s };
            },
            .new_map => .new_map,
            .new_list => .new_list,
            .new_text => .new_text,
        };
    }

    fn decodeStructRef(cur: *[]const u8, agent_count: usize) error{Corrupt}!StructRefWire {
        if (cur.len == 0) return error.Corrupt;
        const tag = cur.*[0];
        cur.* = cur.*[1..];
        return switch (tag) {
            0 => .root,
            1 => .trash,
            2 => blk: {
                const aidx = try getUv(cur);
                if (aidx >= agent_count) return error.Corrupt;
                const seq = try getUv(cur);
                break :blk .{ .node = .{ .agent_idx = @intCast(aidx), .seq = seq } };
            },
            else => error.Corrupt,
        };
    }

    /// Whole-batch causal validation before any graph mutation. `eff_base`
    /// is each batch agent's PROSPECTIVE watermark (its current stored
    /// one, unless this is a bootstrap adopting the batch's own — see
    /// `ObjectDoc.merge`): duplicate/compacted detection must use it
    /// instead of `doc.history`'s not-yet-updated watermark, exactly like
    /// `TextDoc.Decoder.validate`. `batch_head` (this doc's own base
    /// boundary, or the batch's if bootstrapping) is the one compacted
    /// reference a parent ref may name without being separately known or
    /// batch-local — same discipline as `TextDoc`'s.
    ///
    /// "Earlier in batch" is one contiguous seen-range per agent, tracked
    /// incrementally as the single pass proceeds (same shape as
    /// `TextDoc.Decoder.validate`): a causally closed batch has per-agent
    /// ascending contiguous seqs, so the range is exact for honest
    /// encoders; adversarial orderings merely fail to extend it and get
    /// rejected. O(events + parents), not O(events^2).
    fn validate(
        self: *const Decoder,
        gpa: Allocator,
        doc: *const ObjectDoc,
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
            const id: EventId = .{ .agent = aids[ev.agent_idx], .seq = ev.seq };
            const stored = doc.history.agents.items[@intFromEnum(id.agent)].lv_by_seq.items.len;
            const next = eff_base[ev.agent_idx] + stored;
            if (ev.seq < next) continue; // duplicate or compacted
            const contiguous = ev.seq == next or
                (ev.seq > 0 and inRange(seen[ev.agent_idx], ev.seq - 1));
            if (!contiguous) return error.MissingDependency;
            for (self.parentsOf(ev)) |pref| {
                const pid: EventId = .{ .agent = aids[pref.agent_idx], .seq = pref.seq };
                if (doc.history.lvOf(pid) != null) continue;
                if (inRange(seen[pref.agent_idx], pref.seq)) continue;
                if (batch_head) |h| {
                    if (h.agent == pid.agent and h.seq == pid.seq) continue;
                }
                return error.MissingDependency;
            }
        }
        if (self.base) |b| {
            for (b.text_bases) |tb| {
                const id: EventId = .{ .agent = aids[tb.obj.agent_idx], .seq = tb.obj.seq };
                if (doc.history.lvOf(id) != null) continue;
                if (inRange(seen[tb.obj.agent_idx], tb.obj.seq)) continue;
                return error.MissingDependency;
            }
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
