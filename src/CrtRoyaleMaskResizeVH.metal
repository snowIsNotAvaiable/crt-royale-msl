// -----------------------------------------------------------------------------
// CRT-Royale MSL Port -- MASK_RESIZE V/H (slang passes 5 + 6).
//
// Original: crt-royale by TroggleMonkey (GPL v2+).
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang references:
//   crt-royale-mask-resize-vertical.slang
//   crt-royale-mask-resize-horizontal.slang
//   phosphor-mask-resizing.h (helpers; ~700 lines)
//
// Pipeline position:
//   ... small mask LUT (64x64) -> MASK_RESIZE_V (64 x 0.0625*viewport_y)
//                              -> MASK_RESIZE   (0.0625*viewport_x x same_y)
//                              -> consumed by apply-mask (slang pass 7)
//                                 when mask_sample_mode = 0 (slang default).
//
// Why this exists: Slang's apply-mask offers two paths for sampling the
// phosphor mask -- mode 0 reads from a Lanczos-pre-resized tileable LUT
// (this pipeline's output), mode 1 hardware-resamples the original LUT
// with mipmap + anisotropic filtering. Both reach Pass 7's apply-mask code
// path; the only difference is the texture being sampled and the spatial
// frequency content of the mask.
//
// Algorithm: separable Lanczos sinc resampler. Pass 5 resizes the LUT
// vertically into a 64 x 0.0625*viewport_y FBO. Pass 6 then resizes that
// horizontally into the final 0.0625*viewport_x x 0.0625*viewport_y FBO
// (the MASK_RESIZE alias). Both passes:
//   1. Compute mask_resize_tile_size from estimated viewport + tile-min/max
//      constraints. (See phosphor-mask-resizing.h:433-531.)
//   2. Compute tile_uv_wrap from gid / tile_size.
//   3. If tile_uv_wrap > mask_resize_num_tiles, discard (we write zeros).
//   4. Otherwise, fract(tile_uv_wrap) gives the per-tile UV, and
//      tex2D_sinc_1d does the actual Lanczos resampling.
//
// Implementation tradeoffs:
//   - Static 24-sample loop (vs. Slang's dynamic up-to-128). 24 is the
//     tight upper bound for the small (64x64) LUT inputs we use.
//   - Single LUT input (the active mask_type's small LUT, picked by Swift).
//   - mask_specify_num_triads = 0 path only (mask_triad_size_desired
//     drives the tile size).
//
// Status: implemented, runs correctly, produces sensible Lanczos output.
// Not yet bit-exact vs. librashader-cli's per-pass capture -- the discard
// region geometry doesn't line up, because librashader's pass-5/6 output
// at our viewport has only 9 non-zero pixels in a 16x48 FBO (it interprets
// mask_resize_tile_size much smaller than my port). The math for
// get_resized_mask_tile_size + mask_resize_num_tiles depends on a fan of
// derived-settings constants that branch on user-settings.h defines;
// the values librashader uses don't seem to match my hand-traced ones.
// Calibrating this further requires either an instrumented Slang build or
// reading librashader's reflection output to inspect the actual constants.
//
// NOTE: This file is the standalone study mirror; the kernel that actually
// gets built lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // Static slang-derived constants (user-settings.h, user-cgp-constants.h,
    // derived-settings-and-constants.h). Same values used in the production
    // CrtRoyale.metal.
    constexpr constant float2 MR_SRC_LUT_SIZE     = float2(512.0f, 512.0f); // we override at runtime by binding the small 64x64 LUT
    constexpr constant float  MR_TRIADS_PER_TILE  = 8.0f;
    constexpr constant float  MR_MIN_ALLOWED_TILE_SIZE = 16.0f; // = ceil(2*8)
    constexpr constant float  MR_SINC_LOBES       = 3.0f;
    constexpr constant float2 MR_VIEWPORT_SCALE   = float2(0.0625f, 0.0625f);
    constexpr constant float  MR_NUM_TILES        = 2.0f;
    constexpr constant float  MR_GEOM_ASPECT_RATIO = 1.313069909f;
    constexpr constant float  MR_PI               = 3.14159265358979f;
    constexpr constant int    MR_SAMPLES          = 24;

    // get_resized_mask_tile_size: phosphor-mask-resizing.h:433-531.
    // Square LUT -> tile_aspect = (1, 1). Returns floor'd integer tile size.
    inline float2 mr_get_resized_mask_tile_size(
        float2 est_resize_output, float mask_triad_size_desired);

    // Single-axis Lanczos sinc downsample, tiled.
    // Slang ref: downsample_vertical_sinc_tiled / downsample_horizontal_sinc_tiled
    // (phosphor-mask-resizing.h:217-414, with VERTICAL_SINC_RESAMPLE_LOOP_BODY).
    inline float3 mr_downsample_sinc_1d(
        texture2d<float, access::sample> tex, sampler sam,
        float2 src_tex_uv, float2 tex_size,
        float dr, float mag_scale, float tile_size_uv_r, bool vertical);

    kernel void pass_mask_resize_v(
        texture2d<float, access::sample> mask_lut [[ texture(0) ]],
        texture2d<float, access::write>  output   [[ texture(1) ]],
        sampler                          lut_sam  [[ sampler(0) ]],
        uint2                            gid      [[ thread_position_in_grid ]]
    ) {
        // ... see production file. Computes pass_output_tile_size, sets up
        // the per-tile UV, branches on tile_uv_wrap > num_tiles for the
        // slang discard, otherwise dispatches mr_downsample_sinc_1d
        // (vertical=true).
    }

    kernel void pass_mask_resize_h(
        texture2d<float, access::sample> mask_resize_v [[ texture(0) ]],
        texture2d<float, access::write>  output        [[ texture(1) ]],
        sampler                          sam           [[ sampler(0) ]],
        uint2                            gid           [[ thread_position_in_grid ]]
    ) {
        // ... see production file. Mirrors pass 5 but resizes horizontally
        // (vertical=false) and clamps max(tile_uv_wrap.x, tile_uv_wrap.y).
    }

} // namespace crt_royale
