{
  lib,
  stdenv,
  zig_0_16,
  pname ? "stemma",
  version ? "0.7.0",
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

  # The plain nixpkgs zig build: zig.hook runs build/install with the
  # toolchain's default optimize/cpu flags. Nothing else is needed for a
  # normal zig build.
  nativeBuildInputs = [ zig.hook ];

  meta = {
    description = "Event-graph CRDT library with an editor-grade text rope at its core";
    license = lib.licenses.mit;
    platforms = lib.platforms.all;
  };
}
