# Releasing

The version bumps on the **falling edge** of a release: the tagged commit
is the only commit that carries a bare released version. Every commit
after it carries the *next* version with a `-dev` pre-release suffix, so a
build from any untagged tree never claims a released version.

1. Decide the version from the `[Unreleased]` changelog section, per
   semver (pre-1.0: breaking → minor, additive/fix → minor/patch by
   judgment).
2. **Release commit**: retitle `[Unreleased]` → `[X.Y.Z] — YYYY-MM-DD` in
   CHANGELOG.md; set `.version = "X.Y.Z"` in build.zig.zon; update the
   version in README's Status section. Commit as `release: vX.Y.Z`.
3. **Tag** that commit: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
4. **Bump commit**, immediately after: `.version = "X.<Y+1>.0-dev"` in
   build.zig.zon; add a fresh empty `## [Unreleased]` header at the top of
   the changelog. Commit as `start vX.<Y+1>.0-dev`.
5. `git push --follow-tags origin main`.

The `-dev` number is a placeholder, not a promise — if the next release
turns out to be a patch or a major instead, pick the real number at
step 1 and nobody downstream noticed the placeholder.
