//! JsonDoc — a collaborative JSON document over the event graph: the
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
//! Not yet (ledgered): compaction for JsonDoc, identity anchors inside
//! JsonDoc text objects, Peritext-style rich-text marks (representable
//! today as mark objects holding anchor values), incremental persistence.
//! Unification of the TextDoc/JsonDoc doc-core scaffolding is a known
//! cleanup.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const jw = @import("json_walker.zig");
const wire = @import("wire.zig");
const rope_mod = @import("../rope.zig");
const geometry = @import("../geometry.zig");

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
pub const ObjId = jw.ObjId;
const Lv = causal.Lv;
const Graph = jw.Graph;
const Walker = jw.Walker;
const JsonOp = jw.JsonOp;
const ValPayload = jw.ValPayload;
const Str = jw.Str;
const Effect = jw.Effect;
const Rope = rope_mod.Rope;
const Range = geometry.Range;
const Edit = geometry.Edit;
const putUv = wire.putUv;
const getUv = wire.getUv;
const getBytes = wire.getBytes;

const json_magic = "stj\x01";
const version_magic = "stv\x01";
const no_node: u32 = std.math.maxInt(u32);
const root_key = Walker.root_key;

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
    /// A key's value set changed (added, removed, or overwritten) — re-read
    /// via `mapGet`/`mapConflicts`. `key` borrows the doc's string arena
    /// (valid for the doc's lifetime).
    map: struct { obj: ?ObjId, key: []const u8 },
    list_ins: struct { obj: ObjId, index: usize },
    list_del: struct { obj: ObjId, index: usize },
    /// Byte-space edit within a text object; shift that object's anchors.
    text: struct { obj: ObjId, edit: Edit },
};

pub const JsonDoc = struct {
    graph: Graph = .empty,
    agent: ?AgentId = null,
    /// Append-only arena for keys and string values (Str refs point here).
    strings: std.ArrayList(u8) = .empty,
    /// Materialized tree. Node 0 (once created) is the root map.
    nodes: std.ArrayList(Node) = .empty,
    /// Creation-event lv → node index (no_node until materialized).
    node_of: std.ArrayList(u32) = .empty,

    pub const empty: JsonDoc = .{};

    pub const MergeError = Allocator.Error || error{ Corrupt, MissingDependency };
    pub const VersionOrder = causal.VersionOrder;

    const ScalarNode = union(enum) { null_, bool_: bool, int: i64, float: f64, str: Str };
    const TreeVal = struct { set_lv: Lv, node: u32 };
    const TreeSlot = struct { key: Str, values: std.ArrayList(TreeVal) = .empty };
    const Node = union(enum) {
        scalar: ScalarNode,
        map: struct { obj: ?Lv, slots: std.ArrayList(TreeSlot) = .empty },
        list: struct { obj: Lv, elems: std.ArrayList(u32) = .empty },
        text: struct { obj: Lv, rope: Rope = .empty },
    };

    pub fn deinit(self: *JsonDoc, gpa: Allocator) void {
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
        self.strings.deinit(gpa);
        self.graph.deinit(gpa);
        self.* = .{};
    }

    pub fn setAgent(self: *JsonDoc, gpa: Allocator, name: []const u8) Allocator.Error!void {
        self.agent = try self.graph.registerAgent(gpa, name);
    }

    // ObjId values are DOC-LOCAL handles (they embed replica-local agent
    // numbering) — never transport one to another replica. To reference an
    // object across peers, either navigate document structure (paths) or
    // exchange an exported token:

    /// Portable object reference token ("sto" 0x01, name, seq). Caller owns.
    pub fn exportId(self: *const JsonDoc, gpa: Allocator, obj: ObjId) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, "sto\x01");
        const name = self.graph.agentName(obj.agent);
        try putUv(gpa, &out, name.len);
        try out.appendSlice(gpa, name);
        try putUv(gpa, &out, obj.seq);
        return out.toOwnedSlice(gpa);
    }

    /// Resolve a token from `exportId` (possibly minted on another replica)
    /// to this replica's handle. `error.MissingDependency` if we have not
    /// seen the creating event.
    pub fn importId(self: *const JsonDoc, token: []const u8) (error{ Corrupt, MissingDependency })!ObjId {
        var cur = token;
        if (!std.mem.startsWith(u8, cur, "sto\x01")) return error.Corrupt;
        cur = cur[4..];
        const name = try getBytes(&cur, 4096);
        const seq = try getUv(&cur);
        const agent = self.graph.findAgent(name) orelse return error.MissingDependency;
        const id: EventId = .{ .agent = agent, .seq = seq };
        if (self.graph.lvOf(id) == null) return error.MissingDependency;
        return id;
    }

    fn str(self: *const JsonDoc, s: Str) []const u8 {
        return self.strings.items[s.start..][0..s.len];
    }

    fn intern(self: *JsonDoc, gpa: Allocator, bytes: []const u8) Allocator.Error!Str {
        const start: u32 = @intCast(self.strings.items.len);
        try self.strings.appendSlice(gpa, bytes);
        return .{ .start = start, .len = @intCast(bytes.len) };
    }

    fn ensureRoot(self: *JsonDoc, gpa: Allocator) Allocator.Error!void {
        if (self.nodes.items.len == 0) {
            try self.nodes.append(gpa, .{ .map = .{ .obj = null } });
        }
    }

    fn ensureNodeMap(self: *JsonDoc, gpa: Allocator) Allocator.Error!void {
        const n = self.graph.eventCount();
        if (self.node_of.items.len < n) {
            try self.node_of.appendNTimes(gpa, no_node, n - self.node_of.items.len);
        }
    }

    fn nodeOfObjLv(self: *const JsonDoc, obj: Lv) u32 {
        return if (obj == root_key) 0 else self.node_of.items[obj];
    }

    fn resolveObjNode(self: *const JsonDoc, obj: ?ObjId) u32 {
        const id = obj orelse return 0;
        return self.node_of.items[self.graph.lvOf(id).?];
    }

    /// Create the tree node for a freshly applied value payload.
    fn makeValueNode(self: *JsonDoc, gpa: Allocator, val: ValPayload, creation_lv: Lv) Allocator.Error!u32 {
        const idx: u32 = @intCast(self.nodes.items.len);
        const node: Node = switch (val) {
            .null_ => .{ .scalar = .null_ },
            .bool_ => |b| .{ .scalar = .{ .bool_ = b } },
            .int => |v| .{ .scalar = .{ .int = v } },
            .float => |v| .{ .scalar = .{ .float = v } },
            .str => |s| .{ .scalar = .{ .str = s } },
            .new_map => .{ .map = .{ .obj = creation_lv } },
            .new_list => .{ .list = .{ .obj = creation_lv } },
            .new_text => .{ .text = .{ .obj = creation_lv } },
        };
        try self.nodes.append(gpa, node);
        switch (val) {
            .new_map, .new_list, .new_text => self.node_of.items[creation_lv] = idx,
            else => {},
        }
        return idx;
    }

    fn treeSlot(self: *JsonDoc, gpa: Allocator, map_node: u32, key: Str) Allocator.Error!*TreeSlot {
        const m = &self.nodes.items[map_node].map;
        for (m.slots.items) |*s| {
            if (std.mem.eql(u8, self.str(s.key), self.str(key))) return s;
        }
        try m.slots.append(gpa, .{ .key = key });
        return &m.slots.items[m.slots.items.len - 1];
    }

    // ── Local editing ───────────────────────────────────────────────────

    fn payloadFrom(self: *JsonDoc, gpa: Allocator, v: Value) Allocator.Error!ValPayload {
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
    pub fn mapSet(self: *JsonDoc, gpa: Allocator, obj: ?ObjId, key: []const u8, val: Value) Allocator.Error!?ObjId {
        const agent = self.agent.?;
        try self.ensureRoot(gpa);
        const map_node = self.resolveObjNode(obj);
        assert(self.nodes.items[map_node] == .map);
        const key_str = try self.intern(gpa, key);
        const payload = try self.payloadFrom(gpa, val);
        const lv = try self.graph.addLocal(gpa, agent, .{ .map_set = .{ .obj = obj, .key = key_str, .val = payload } });
        try self.ensureNodeMap(gpa);
        const value_node = try self.makeValueNode(gpa, payload, lv);
        const slot = try self.treeSlot(gpa, map_node, key_str);
        slot.values.clearRetainingCapacity(); // local set overwrites all seen
        try slot.values.append(gpa, .{ .set_lv = lv, .node = value_node });
        return switch (payload) {
            .new_map, .new_list, .new_text => self.graph.idOf(lv),
            else => null,
        };
    }

    pub fn mapDelete(self: *JsonDoc, gpa: Allocator, obj: ?ObjId, key: []const u8) Allocator.Error!void {
        const agent = self.agent.?;
        try self.ensureRoot(gpa);
        const map_node = self.resolveObjNode(obj);
        assert(self.nodes.items[map_node] == .map);
        const key_str = try self.intern(gpa, key);
        _ = try self.graph.addLocal(gpa, agent, .{ .map_del = .{ .obj = obj, .key = key_str } });
        try self.ensureNodeMap(gpa);
        const slot = try self.treeSlot(gpa, map_node, key_str);
        slot.values.clearRetainingCapacity();
    }

    /// Insert into list `obj` at `index`. Returns the created object's id
    /// for `.map`/`.list`/`.text` values.
    pub fn listInsert(self: *JsonDoc, gpa: Allocator, obj: ObjId, index: usize, val: Value) Allocator.Error!?ObjId {
        const agent = self.agent.?;
        try self.ensureRoot(gpa);
        const list_node = self.resolveObjNode(obj);
        const l = &self.nodes.items[list_node].list;
        assert(index <= l.elems.items.len);
        const payload = try self.payloadFrom(gpa, val);
        const lv = try self.graph.addLocal(gpa, agent, .{ .list_ins = .{ .obj = obj, .pos = index, .val = payload } });
        try self.ensureNodeMap(gpa);
        const value_node = try self.makeValueNode(gpa, payload, lv);
        try self.nodes.items[list_node].list.elems.insert(gpa, index, value_node);
        return switch (payload) {
            .new_map, .new_list, .new_text => self.graph.idOf(lv),
            else => null,
        };
    }

    pub fn listDelete(self: *JsonDoc, gpa: Allocator, obj: ObjId, index: usize) Allocator.Error!void {
        const agent = self.agent.?;
        const list_node = self.resolveObjNode(obj);
        const l = &self.nodes.items[list_node].list;
        assert(index < l.elems.items.len);
        _ = try self.graph.addLocal(gpa, agent, .{ .list_del = .{ .obj = obj, .pos = index } });
        _ = self.nodes.items[list_node].list.elems.orderedRemove(index);
    }

    /// Insert UTF-8 `content` into text object `obj` at `byte_offset`.
    /// Same contract as `Rope.insert`.
    pub fn textInsert(self: *JsonDoc, gpa: Allocator, obj: ObjId, byte_offset: usize, content: []const u8) Allocator.Error!Edit {
        const agent = self.agent.?;
        const text_node = self.resolveObjNode(obj);
        const t = &self.nodes.items[text_node].text;
        const edit: Edit = .{ .offset = byte_offset, .removed = 0, .inserted = content.len };
        if (content.len == 0) return edit;
        const scalar_pos = t.rope.offsetToScalar(byte_offset);
        var i: u64 = 0;
        var it = (std.unicode.Utf8View.init(content) catch unreachable).iterator();
        while (it.nextCodepoint()) |ch| : (i += 1) {
            _ = try self.graph.addLocal(gpa, agent, .{ .text_ins = .{ .obj = obj, .pos = scalar_pos + i, .ch = ch } });
        }
        _ = try self.nodes.items[text_node].text.rope.insert(gpa, byte_offset, content);
        return edit;
    }

    pub fn textDelete(self: *JsonDoc, gpa: Allocator, obj: ObjId, range: Range) Allocator.Error!Edit {
        const agent = self.agent.?;
        const text_node = self.resolveObjNode(obj);
        const t = &self.nodes.items[text_node].text;
        const edit: Edit = .{ .offset = range.start, .removed = range.len(), .inserted = 0 };
        if (range.isEmpty()) return edit;
        const scalar_start = t.rope.offsetToScalar(range.start);
        const scalar_count = t.rope.offsetToScalar(range.end) - scalar_start;
        for (0..scalar_count) |_| {
            _ = try self.graph.addLocal(gpa, agent, .{ .text_del = .{ .obj = obj, .pos = scalar_start } });
        }
        _ = try self.nodes.items[text_node].text.rope.delete(gpa, range);
        return edit;
    }

    // ── Reads ───────────────────────────────────────────────────────────

    pub fn root(self: *const JsonDoc) ValueRef {
        return .{ .doc = self, .node = 0 };
    }

    pub const ValueRef = struct {
        doc: *const JsonDoc,
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
        pub fn objId(self: ValueRef) ?ObjId {
            const n = self.nodePtr() orelse return null;
            const lv: ?Lv = switch (n.*) {
                .map => |m| m.obj,
                .list => |l| l.obj,
                .text => |t| t.obj,
                .scalar => unreachable, // scalars have no identity
            };
            return if (lv) |v| self.doc.graph.idOf(v) else null;
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
                if (self.doc.setOrder(best.set_lv, v.set_lv) == .lt) best = v;
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

    fn setOrder(self: *const JsonDoc, a: Lv, b: Lv) std.math.Order {
        const ia = self.graph.idOf(a);
        const ib = self.graph.idOf(b);
        const name_order = std.mem.order(u8, self.graph.agentName(ia.agent), self.graph.agentName(ib.agent));
        if (name_order != .eq) return name_order;
        return std.math.order(ia.seq, ib.seq);
    }

    /// Canonical JSON dump (winner-only, keys sorted, text objects as
    /// strings). Caller owns. Conflicts are invisible here — inspect them
    /// via `mapConflictCount`.
    pub fn toJson(self: *const JsonDoc, gpa: Allocator) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try self.dumpValue(gpa, &out, self.root());
        return out.toOwnedSlice(gpa);
    }

    fn dumpValue(self: *const JsonDoc, gpa: Allocator, out: *std.ArrayList(u8), v: ValueRef) Allocator.Error!void {
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

    pub fn version(self: *const JsonDoc, gpa: Allocator) Allocator.Error![]u8 {
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

    fn decodeVersion(
        self: *const JsonDoc,
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
            const name = try getBytes(&cur, 4096);
            if (name.len == 0) return error.Corrupt;
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

    pub fn compareVersions(self: *const JsonDoc, gpa: Allocator, a_token: []const u8, b_token: []const u8) MergeError!VersionOrder {
        var a: std.ArrayList(Lv) = .empty;
        defer a.deinit(gpa);
        try self.decodeVersion(gpa, a_token, true, &a);
        var b: std.ArrayList(Lv) = .empty;
        defer b.deinit(gpa);
        try self.decodeVersion(gpa, b_token, true, &b);
        return self.graph.compareFrontiers(gpa, a.items, b.items);
    }

    pub fn eventsSince(self: *const JsonDoc, gpa: Allocator, remote_version: []const u8) (Allocator.Error || error{Corrupt})![]u8 {
        var known: std.ArrayList(Lv) = .empty;
        defer known.deinit(gpa);
        self.decodeVersion(gpa, remote_version, false, &known) catch |e| switch (e) {
            error.MissingDependency => unreachable,
            else => |err| return err,
        };
        var missing = try self.graph.missingFrom(gpa, known.items);
        defer missing.deinit(gpa);
        return self.encodeEvents(gpa, missing.items);
    }

    pub fn serialize(self: *const JsonDoc, gpa: Allocator) Allocator.Error![]u8 {
        var missing = try self.graph.missingFrom(gpa, &.{});
        defer missing.deinit(gpa);
        return self.encodeEvents(gpa, missing.items);
    }

    pub fn open(gpa: Allocator, bytes: []const u8) MergeError!JsonDoc {
        var doc: JsonDoc = .empty;
        errdefer doc.deinit(gpa);
        const changes = try doc.merge(gpa, bytes);
        gpa.free(changes);
        return doc;
    }

    // ── Merge ───────────────────────────────────────────────────────────

    /// Integrate encoded remote events; returns the change stream (caller
    /// owns). Same atomic-reject semantics as `TextDoc.merge`.
    pub fn merge(self: *JsonDoc, gpa: Allocator, bytes: []const u8) MergeError![]Change {
        var dec = try Decoder.init(gpa, bytes);
        defer dec.deinit(gpa);

        const aids = try gpa.alloc(AgentId, dec.names.items.len);
        defer gpa.free(aids);
        for (dec.names.items, aids) |name, *aid| {
            aid.* = try self.graph.registerAgent(gpa, name);
        }
        try dec.validate(self, aids);

        // Graph phase with wholesale rollback (including interned strings).
        var effects: std.ArrayList(Effect) = .empty;
        defer effects.deinit(gpa);
        const any_new = try self.graphPhase(gpa, &dec, aids, &effects);
        if (!any_new) return try gpa.alloc(Change, 0);

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
                    const slot = try self.treeSlot(gpa, map_node, e.key);
                    try slot.values.append(gpa, .{ .set_lv = e.set_lv, .node = value_node });
                    try changes.append(gpa, .{ .map = .{ .obj = self.objIdOf(e.obj), .key = self.str(e.key) } });
                },
                .map_remove => |e| {
                    const map_node = self.nodeOfObjLv(e.obj orelse root_key);
                    const slot = try self.treeSlot(gpa, map_node, e.key);
                    for (slot.values.items, 0..) |v, i| {
                        if (v.set_lv == e.set_lv) {
                            _ = slot.values.orderedRemove(i);
                            break;
                        }
                    }
                    try changes.append(gpa, .{ .map = .{ .obj = self.objIdOf(e.obj), .key = self.str(e.key) } });
                },
                .list_ins => |e| {
                    const list_node = self.nodeOfObjLv(e.obj);
                    const value_node = try self.makeValueNode(gpa, e.val, e.lv);
                    try self.nodes.items[list_node].list.elems.insert(gpa, @intCast(e.index), value_node);
                    try changes.append(gpa, .{ .list_ins = .{ .obj = self.graph.idOf(e.obj), .index = @intCast(e.index) } });
                },
                .list_del => |e| {
                    const list_node = self.nodeOfObjLv(e.obj);
                    _ = self.nodes.items[list_node].list.elems.orderedRemove(@intCast(e.index));
                    try changes.append(gpa, .{ .list_del = .{ .obj = self.graph.idOf(e.obj), .index = @intCast(e.index) } });
                },
                .text_ins => |e| {
                    const text_node = self.nodeOfObjLv(e.obj);
                    const t = &self.nodes.items[text_node].text;
                    const off = t.rope.scalarToOffset(e.pos);
                    const len = std.unicode.utf8Encode(e.ch, &buf) catch unreachable;
                    _ = try self.nodes.items[text_node].text.rope.insert(gpa, off, buf[0..len]);
                    try appendTextChange(gpa, &changes, self.graph.idOf(e.obj), .{ .offset = off, .removed = 0, .inserted = len });
                },
                .text_del => |e| {
                    const text_node = self.nodeOfObjLv(e.obj);
                    const t = &self.nodes.items[text_node].text;
                    const start = t.rope.scalarToOffset(e.pos);
                    const end = t.rope.scalarToOffset(e.pos + 1);
                    _ = try self.nodes.items[text_node].text.rope.delete(gpa, .{ .start = start, .end = end });
                    try appendTextChange(gpa, &changes, self.graph.idOf(e.obj), .{ .offset = start, .removed = end - start, .inserted = 0 });
                },
            }
        }
        return changes.toOwnedSlice(gpa);
    }

    fn objIdOf(self: *const JsonDoc, obj: ?Lv) ?ObjId {
        return if (obj) |lv| self.graph.idOf(lv) else null;
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

    fn graphPhase(
        self: *JsonDoc,
        gpa: Allocator,
        dec: *const Decoder,
        aids: []const AgentId,
        effects: *std.ArrayList(Effect),
    ) MergeError!bool {
        const pre_events = self.graph.events.items.len;
        const pre_pool = self.graph.parents_pool.items.len;
        const pre_strings = self.strings.items.len;
        const pre_frontier = try gpa.dupe(Lv, self.graph.frontier.items);
        defer gpa.free(pre_frontier);
        const pre_seq_lens = try gpa.alloc(usize, self.graph.agents.items.len);
        defer gpa.free(pre_seq_lens);
        for (self.graph.agents.items, pre_seq_lens) |a, *len| len.* = a.lv_by_seq.items.len;
        errdefer {
            self.graph.events.items.len = pre_events;
            self.graph.parents_pool.items.len = pre_pool;
            self.strings.items.len = pre_strings;
            for (self.graph.agents.items, 0..) |*a, i| {
                a.lv_by_seq.items.len = if (i < pre_seq_lens.len) pre_seq_lens[i] else 0;
            }
            self.graph.frontier.clearRetainingCapacity();
            self.graph.frontier.appendSliceAssumeCapacity(pre_frontier);
        }

        const first_new: Lv = @intCast(self.graph.eventCount());
        var any_new = false;
        for (dec.events.items) |ev| {
            const id: EventId = .{ .agent = aids[ev.agent_idx], .seq = ev.seq };
            if (self.graph.isKnown(id)) continue;
            var parent_lvs: std.ArrayList(Lv) = .empty;
            defer parent_lvs.deinit(gpa);
            for (dec.parentsOf(ev)) |pref| {
                const pid: EventId = .{ .agent = aids[pref.agent_idx], .seq = pref.seq };
                try parent_lvs.append(gpa, self.graph.lvOf(pid).?);
            }
            const op = try self.internOp(gpa, ev, aids);
            _ = try self.graph.add(gpa, id, parent_lvs.items, op);
            any_new = true;
        }
        if (!any_new) return false;

        var w = Walker.init(&self.graph, self.strings.items);
        defer w.deinit(gpa);
        try w.replayAll(gpa, first_new, effects);
        return true;
    }

    /// Build the stored op from a decoded one: intern strings, resolve
    /// object refs from batch (agent_idx, seq) to EventIds.
    fn internOp(self: *JsonDoc, gpa: Allocator, ev: Decoder.Event, aids: []const AgentId) MergeError!JsonOp {
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
        };
    }

    fn internVal(self: *JsonDoc, gpa: Allocator, v: Decoder.RawVal) Allocator.Error!ValPayload {
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
    // "stj" 0x01: uv agent_count, per agent (uv name_len, name);
    // uv event_count, per event: uv agent_idx, uv seq, uv parent_count,
    // parents (uv agent_idx, uv seq); u8 op tag; obj ref (u8 0 = root,
    // 1 + uv agent_idx + uv seq); op-specific fields. Ints are zigzag'd.

    const OpTag = enum(u8) { map_set = 0, map_del = 1, list_ins = 2, list_del = 3, text_ins = 4, text_del = 5 };
    const ValTag = enum(u8) { null_ = 0, false_ = 1, true_ = 2, int = 3, float = 4, str = 5, new_map = 6, new_list = 7, new_text = 8 };

    fn encodeEvents(self: *const JsonDoc, gpa: Allocator, lvs: []const Lv) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, json_magic);

        var table: std.ArrayList(AgentId) = .empty;
        defer table.deinit(gpa);
        for (lvs) |lv| {
            try tableAdd(gpa, &table, self.graph.idOf(lv).agent);
            for (self.graph.parentsOf(lv)) |p| try tableAdd(gpa, &table, self.graph.idOf(p).agent);
            if (opObj(self.graph.opOf(lv))) |o| try tableAdd(gpa, &table, o.agent);
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
            const op = self.graph.opOf(lv);
            try out.append(gpa, @intFromEnum(@as(OpTag, switch (op) {
                .map_set => .map_set,
                .map_del => .map_del,
                .list_ins => .list_ins,
                .list_del => .list_del,
                .text_ins => .text_ins,
                .text_del => .text_del,
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
            }
        }
        return out.toOwnedSlice(gpa);
    }

    fn opObj(op: JsonOp) ?ObjId {
        return switch (op) {
            .map_set => |m| m.obj,
            .map_del => |m| m.obj,
            .list_ins => |l| l.obj,
            .list_del => |l| l.obj,
            .text_ins => |x| x.obj,
            .text_del => |x| x.obj,
        };
    }

    fn encodeVal(self: *const JsonDoc, gpa: Allocator, out: *std.ArrayList(u8), v: ValPayload) Allocator.Error!void {
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

    fn tableAdd(gpa: Allocator, table: *std.ArrayList(AgentId), aid: AgentId) Allocator.Error!void {
        for (table.items) |x| if (x == aid) return;
        try table.append(gpa, aid);
    }

    fn tableIndexOf(table: []const AgentId, aid: AgentId) usize {
        for (table, 0..) |x, i| if (x == aid) return i;
        unreachable;
    }

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
        };

        names: std.ArrayList([]const u8) = .empty,
        events: std.ArrayList(Event) = .empty,
        parents_pool: std.ArrayList(ObjRef) = .empty,

        fn parentsOf(self: *const Decoder, ev: Event) []const ObjRef {
            return self.parents_pool.items[ev.parents_start..][0..ev.parents_len];
        }

        fn deinit(self: *Decoder, gpa: Allocator) void {
            self.names.deinit(gpa);
            self.events.deinit(gpa);
            self.parents_pool.deinit(gpa);
        }

        fn init(gpa: Allocator, bytes: []const u8) JsonDoc.MergeError!Decoder {
            var self: Decoder = .{};
            errdefer self.deinit(gpa);
            var cur: []const u8 = bytes;
            if (!std.mem.startsWith(u8, cur, json_magic)) return error.Corrupt;
            cur = cur[json_magic.len..];

            const agent_count = try getUv(&cur);
            if (agent_count > 1 << 20) return error.Corrupt;
            for (0..agent_count) |_| {
                const name = try getBytes(&cur, 4096);
                if (name.len == 0) return error.Corrupt;
                try self.names.append(gpa, name);
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
                if (tag_byte > 5) return error.Corrupt;
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
                // Root refs are only legal for map ops.
                if (ev.obj == null and ev.op_tag != .map_set and ev.op_tag != .map_del) return error.Corrupt;
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

        fn validate(self: *const Decoder, doc: *const JsonDoc, aids: []const AgentId) error{MissingDependency}!void {
            for (self.events.items, 0..) |ev, i| {
                const id: EventId = .{ .agent = aids[ev.agent_idx], .seq = ev.seq };
                if (doc.graph.isKnown(id)) continue;
                const next = doc.graph.nextSeq(id.agent);
                const contiguous = ev.seq == next or
                    (ev.seq > 0 and self.seenEarlier(i, ev.agent_idx, ev.seq - 1));
                if (!contiguous) return error.MissingDependency;
                for (self.parentsOf(ev)) |pref| {
                    const pid: EventId = .{ .agent = aids[pref.agent_idx], .seq = pref.seq };
                    if (doc.graph.lvOf(pid) != null) continue;
                    if (self.seenEarlier(i, pref.agent_idx, pref.seq)) continue;
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

test {
    std.testing.refAllDecls(JsonDoc);
}
