let
  npins = import ./npins;

  mkPackages = pkgs: {
    # The derivation fileset-scopes its own src (see nix/stemma.nix), so no
    # src plumbing is needed here.
    stemma = pkgs.callPackage ./nix/stemma.nix { };
  };

  # Scoped against `final` so packages can reference each other; lazy, so no
  # infinite recursion. Consumers apply this overlay to their own pkgs to get
  # stemma's packages by name.
  overlay = final: _prev: mkPackages final;
in
{
  sources ? npins,
  nixpkgs ? sources.nixpkgs,
  pkgs ? import nixpkgs { },
  ...
}:
let
  finalPkgs = pkgs.extend overlay;
in
{
  packages = mkPackages finalPkgs;
  inherit overlay;
  shell = import ./shell.nix { pkgs = finalPkgs; };
  default = finalPkgs.stemma;
}
