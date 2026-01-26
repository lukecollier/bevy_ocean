# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Bevy app (`main.rs`), shared plugins (`cloud_plugin.rs`, `sky_plugin.rs`, etc.), and the ocean-specific modules inside `src/ocean/`.
- Visual assets live under `assets/`, textures under `src/textures/`, and runtime samples or exports under `public/`.
- WGSL shaders are collected in `src/shaders/`; each shader targets a specific pass (for example, `waves_data_merge.wgsl` handles mipmap output).
- `Cargo.toml` defines the Bevy workspace, and `WATER_PLAN.md` documents our current roadmap/ideas.

## Build, Test, and Development Commands
- `cargo build` — compiles the entire Bevy application (use `--release` for performance profiling).
- `cargo run --release` — runs the rendered scene through Bevy with the optimized shader pipeline.
- `cargo test` — runs all Rust unit and integration tests; useful before pushing a branch.
- `cargo fmt` and `cargo clippy` — enforce the Rust style and lint rules that keep the shared code clean.
- Shader edits only? Reload via Bevy’s hot-reload or rerun `cargo run` after editing WGSL under `src/shaders/`.

## Coding Style & Naming Conventions
- Follow `rustfmt` for Rust files; `snake_case` for functions/variables, `CamelCase` for structs/enums, and uppercase `const` names when appropriate.
- Keep shader helpers in `snake_case`, place shared utilities near the top of their WGSL file, and group I/O bindings together for clarity.
- Keep plugin modules focused: avoid monolithic files by splitting ocean, sky, and day-night responsibilities into their respective plugin files under `src/`.
- Inline comments should explain “why” when you deviate from straightforward logic (e.g., why a shader samples neighbors in a particular order).

## Testing Guidelines
- Run `cargo test` to cover Rust logic; there are currently no automated shader tests, so manually verify visual artifacts after builds.
- Name test functions with `snake_case` and describe behavior (`wave_data_merge_handles_mip`).
- When fixing graphical bugs, capture before/after screenshots or notes and mention them in PRs so reviewers can reproduce regressions.

## Commit & Pull Request Guidelines
- Follow observed commit style (`feat:`, `chore:`, `fix:` prefixes), keeping messages short but descriptive (e.g., `feat: add mip-aware blur pass`).
- PRs should explain what changed, why, and how it was verified; link to relevant issues when available and attach before/after visuals for render changes.
- Mention `cargo fmt` and `cargo test` results in the description so reviewers know development checks were run.
- Keep PR size focused—split large refactors into logical chunks to ease shader review and manual render testing.
