const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Public module: `stemma` ──
    const stemma_mod = b.addModule("stemma", .{
        .root_source_file = b.path("src/stemma/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── Static library artifact (installed by `zig build`) ──
    const lib = b.addLibrary(.{
        .name = "stemma",
        .root_module = stemma_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // ── Tests ──
    const unit_tests = b.addTest(.{ .root_module = stemma_mod });
    const run_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // ── F3 validation sketch (exploratory, additive; app/weft's
    // north-star-plan.md §5 F3) ──
    // Deliberately its own module, NOT reachable from `stemma_mod`/root.zig:
    // it validates parent-register + fractional-order-key structure before
    // any client depends on it, consuming only causal.zig's EventGraph
    // substrate. Wired into `zig build test` here so the gate is real
    // without touching the public module graph.
    const sketch_mod = b.createModule(.{
        .root_source_file = b.path("src/stemma/collab/structure_sketch.zig"),
        .target = target,
        .optimize = optimize,
    });
    const sketch_tests = b.addTest(.{ .root_module = sketch_mod });
    const run_sketch_tests = b.addRunArtifact(sketch_tests);
    test_step.dependOn(&run_sketch_tests.step);

    // ── Benchmarks (always ReleaseFast: Debug numbers are lies) ──
    const bench_stemma = b.createModule(.{
        .root_source_file = b.path("src/stemma/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("dev/bench/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("stemma", bench_stemma);
    const bench_exe = b.addExecutable(.{
        .name = "stemma-bench",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    if (b.args) |args| run_bench.addArgs(args);
    const bench_step = b.step("bench", "Run benchmarks (pass a filter substring after --)");
    bench_step.dependOn(&run_bench.step);

    // ── WASM target (browser peers speak the same wire protocol) ──
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/stemma/root.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .wasi }),
        .optimize = .ReleaseSmall,
    });
    const wasm_lib = b.addLibrary(.{
        .name = "stemma",
        .root_module = wasm_mod,
        .linkage = .static,
    });
    const wasm_step = b.step("wasm", "Build the library for wasm32-wasi");
    wasm_step.dependOn(&b.addInstallArtifact(wasm_lib, .{ .dest_dir = .{ .override = .{ .custom = "wasm" } } }).step);
}
