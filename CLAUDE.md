# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo run                  # Debug build and run (with UI controls)
cargo run --release        # Optimized release build
cargo build                # Debug build only
cargo check                # Fast type checking without full compilation
```

Note: Dev builds use `opt-level = 1` with dependencies at `opt-level = 3` for faster iteration. The project uses Rust edition 2024.

## Architecture Overview

This is a **GPU-accelerated FFT ocean simulation** built with Bevy 0.17. The simulation uses the Phillips spectrum and implements a multi-cascade system for realistic wave rendering at multiple scales.

### Core Pipeline Flow

```
Initial Spectrum → Time Evolution → Inverse FFT → Data Merge → Foam → Mipmaps → Render
```

Each frame dispatches these compute pipelines for 3 independent cascades:
- **Cascade 0** (500m scale): Large swells
- **Cascade 1** (85m scale): Medium waves, fades in under 2000m
- **Cascade 2** (10m scale): Fine details, fades in under 300m

### Key Module Structure

**Main Plugin** (`src/ocean_plugin.rs`):
- `OceanPlugin`: Registers materials, render graph node, and synchronization systems
- `OceanMaterial`: Extended PBR material with displacement, derivatives, and foam textures
- `OceanNode`: Render graph node managing compute dispatch and initialization state machine

**Cascade Orchestration** (`src/ocean/`):
- `ocean_cascade.rs`: Manages 3 cascades with different frequency ranges
- `ocean_surface.rs`: Single cascade pipeline orchestration
- `ocean_parameters.rs`: Spectrum configuration (wind, depth, JONSWAP parameters)

**GPU Pipelines** (`src/ocean/pipelines/`):
- `initial_spectrum_pipeline.rs`: Phillips spectrum generation with Donelan-Banner spreading
- `time_dependent_spectrum_pipeline.rs`: Wave animation via phase rotation
- `fft.rs`: Cooley-Tukey inverse FFT (8 horizontal + 8 vertical passes)
- `waves_data_merge_pipeline.rs`: Combines FFT outputs, computes Jacobian for foam
- `foam_persistence_pipeline.rs`: Exponential decay foam accumulation
- `generate_mipmaps_pipeline.rs`: LOD mipmap generation

**Shaders** (`assets/shaders/`):
- `.wgsl` files correspond 1:1 with pipeline modules
- `ocean_shader.wgsl`: Vertex displacement sampling + PBR fragment with foam/SSS

### Texture Layout

All simulation textures are 256×256 RGBA32Float:
- **Displacement**: XYZ wave offset + Jacobian determinant (4 mip levels)
- **Derivatives**: Surface slopes for normal mapping (4 mip levels)
- **Foam Persistence**: Single-channel foam accumulation

### Render Graph Integration

The `OceanNode` uses a state machine:
1. `Init(0)`: Wait for pipeline creation
2. `Init(1)`: Run initial spectrum + FFT precompute (once)
3. `Run`: Execute per-frame dispatch loop

### UI Parameters (runtime adjustable)

The app exposes 8 sliders via Bevy Feathers UI that sync to shader uniforms each frame:
- Displacement scale, normal strength, foam threshold/multiplier
- Foam tile scale, roughness, light intensity, SSS intensity

## Key Considerations

- **Bevy 0.17**: Uses experimental features (`experimental_bevy_feathers`, `meshlet_processor`)
- **Workgroup size**: 16×16 (256 threads) for most compute shaders
- **FFT passes**: 16 total per cascade per frame (log₂(256) × 2 dimensions)
- **Linting**: Allows `too_many_arguments` and `type_complexity` (idiomatic Bevy patterns)
