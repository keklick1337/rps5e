## What and why

<!-- What the change does and the root cause it fixes. Link the issue: Fixes #123 -->

## How it was tested

<!-- Unit tests added or run; for games: which game and scene, before/after frames or frame times. -->

## Checklist

- [ ] `cargo build --release` has no warnings
- [ ] `cargo test --release` passes (and `make test-gpu` for GPU changes)
- [ ] The Windows build still compiles (`cargo build --release --target x86_64-pc-windows-gnu`), or the change is Linux-only code
- [ ] New source files carry the project header (see CONTRIBUTING.md)
- [ ] No game-specific workarounds; no copied proprietary code
