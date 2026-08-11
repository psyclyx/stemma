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
}
