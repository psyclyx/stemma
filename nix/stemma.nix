{
  lib,
  stdenv,
  zig_0_16,
  src ? ../.,
  pname ? "stemma",
  version ? "0.1.0",
  optimize ? "fast",
  cpu ? "baseline",
}:

let
  zig = zig_0_16;

  cleanSrc = lib.cleanSourceWith {
    inherit src;
    filter =
      path: type:
      let
        name = builtins.baseNameOf path;
      in
      !(builtins.elem name [
        ".direnv"
        ".worktrees"
        ".zig-cache"
        "zig-out"
      ])
      && lib.cleanSourceFilter path type;
  };
in
stdenv.mkDerivation {
  inherit pname version;
  src = cleanSrc;

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
