{
  lib,
  stdenv,
  zig_0_16,
  pname ? "stemma",
  version ? "0.7.0",
  optimize ? "fast",
  cpu ? "baseline",
}:

let
  zig = zig_0_16;
in
stdenv.mkDerivation {
  inherit pname version;
  # Only the files the build actually consumes: the build graph and the
  # library sources. Entry points (default.nix, shell.nix), npins/, docs,
  # and dev/ benchmarks are not package inputs, so editing them must not
  # churn the source hash.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../build.zig
      ../build.zig.zon
      ../src
    ];
  };

  # zig.hook drives `zig build` (configure/build/install phases) using the
  # pinned Zig from nixpkgs. No C deps, so no buildInputs.
  nativeBuildInputs = [ zig.hook ];

  zigBuildFlags = [
    "--release=${optimize}"
    "-Dcpu=${cpu}"
  ];

  # The Zig `test` step is exercised via `nix-shell --run 'zig build test'`
  # and CI rather than baked into the derivation's check phase.
  dontUseZigCheck = true;
  dontSetZigDefaultFlags = true;

  meta = {
    description = "Event-graph CRDT library with an editor-grade text rope at its core";
    license = lib.licenses.mit;
    platforms = lib.platforms.all;
  };
}
