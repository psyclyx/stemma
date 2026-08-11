let
  npins = import ./npins;

  mkPackages = pkgs: {
    # Pass src explicitly: nix/stemma.nix has a `src` formal, and without this
    # callPackage would fill it from pkgs.src (a throwing alias). The
    # derivation cleans the tree itself.
    stemma = pkgs.callPackage ./nix/stemma.nix { src = ./.; };
  };

  # Scoped against `final` so packages can reference each other; lazy, so no
  # infinite recursion. Consumers apply this overlay to their own pkgs to get
  # stemma's packages by name.
  overlay = final: _prev: mkPackages final;
in
{
  nixpkgs ? npins.nixpkgs,
  pkgs ? import nixpkgs { },
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
