{
  pkgs ? import (import ./npins).nixpkgs { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    zig_0_16

    # Not build inputs — the library needs none of these. `treefmt` (config:
    # treefmt.toml) drives nixfmt + `zig fmt`; the pre-commit hook
    # (hooks/pre-commit, wired via .envrc) runs `treefmt --fail-on-change`.
    treefmt
    nixfmt
    deadnix # dead Nix bindings
    statix # Nix anti-pattern lints
    nixd # LSP: Nix
    zls # LSP: Zig (0.16, matches zig_0_16)
    marksman # LSP: Markdown
    taplo # LSP + formatter: TOML
    shellcheck # lints hooks/pre-commit
  ];
}
