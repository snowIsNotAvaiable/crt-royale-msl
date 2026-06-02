// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Pass 3: Apply Phosphor Mask
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang reference: crt-royale-scanlines-horizontal-apply-mask.{slang,h}
// Slang pass index: 7   (in our pipeline this is Pass 3, since we have not yet
//                        ported Slang Passes 2..6 -- BLOOM_APPROX, halation
//                        Blur9 V/H, Mask Resize V/H).
//
// What this pass does:
//   For each output pixel, sample the upstream vertical-scanline texture and
//   multiply by a phosphor mask. The mask gates light into R/G/B subpixel
//   bands, simulating the discrete phosphor stripes / dots of a CRT.
//
// Two paths, controlled by uniforms.mask_lut_enabled:
//
//   1. LUT path (slang-equivalent, mask_lut_enabled = 1):
//      Bind mask_grille_texture_large at texture(2). Sample at
//      mask_tex_uv = (gid + 0.5) / mask_resize_tile_size where
//      mask_resize_tile_size = mask_triads_per_tile (=8) * mask_triad_size.
//      This is the slang hardware-resample branch (mask_sample_mode > 0.5):
//      convert_phosphor_tile_uv_wrap_to_tex_uv returns tile_uv_wrap as-is.
//
//      The LUT is mipmapped and sampled with linear+repeat. Compute shaders
//      have no implicit ddx/ddy, so we pass analytical UV gradients via
//      gradient2d() so the GPU can pick the right mipmap LOD (the slang
//      fragment shader gets this for free from rasterization).
//
//      Validated against `librashader-cli --params mask_sample_mode_desired=1`:
//      mean ΔE2000 = 0.36..2.89 across the test suite, SSIM 0.96..0.99.
//
//   2. Procedural fallback (mask_lut_enabled = 0):
//      Generate an aperture grille on the fly. Used when no LUT texture is
//      bound. Sharper-looking output but no slang equivalence.
//
// Static-config simplifications still in place relative to the slang ref:
//   - No horizontal beam-filter shaping (sample_rgb_scanline_horizontal):
//     direct linear sample of the input. Equivalent to beam_horiz_filter
//     defaulting to a delta function rather than Quilez/Lanczos2.
//   - convergence_offset_x_{r,g,b} = 0 (no chromatic aberration).
//   - halation_weight = 0 (this is the slang default too, but explicit here).
//   - No PHOSPHOR_BLOOM_FAKE.
//   - Only mask_type = 0 (aperture grille) currently shipped. Slot/shadow
//     LUTs exist in vendor/slang-shaders -- enabling them is a matter of
//     loading the other PNGs and adding a mask_type runtime switch.
//
// The mask_amplify and undim factors that slang's pass 11 applies live in
// our pass_final_encode for now (collapsed because pass 11 isn't ported).
// Pass 3 itself outputs the "dim" scanline * mask product unmodified --
// this matches slang pass 7's output and is what the validation pipeline
// compares against.
//
// NOTE: This file is the standalone study mirror; the kernel that gets built
// lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // The full Uniforms struct lives in CrtRoyale.metal. Relevant fields:
    //   uint  mask_type;            // 0 = aperture grille
    //   float mask_triad_size;      // output pixels per RGB triad
    //   float mask_amplify;         // ~ 4.81 for grille15; applied in pass11
    //   uint  mask_lut_enabled;     // 0 = procedural, 1 = sample LUT

    kernel void pass3_apply_mask(
        texture2d<float, access::sample> input    [[ texture(0) ]],
        texture2d<float, access::write>  output   [[ texture(1) ]],
        texture2d<float, access::sample> mask_lut [[ texture(2) ]],
        constant Uniforms&               u        [[ buffer(0)  ]],
        sampler                          sam      [[ sampler(0) ]],
        sampler                          lut_sam  [[ sampler(1) ]],
        uint2                            gid      [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());
        float3 scanline_color = input.sample(sam, uv).rgb;

        float3 mask;
        if (u.mask_lut_enabled != 0u) {
            // Slang ref: crt-royale-scanlines-horizontal-apply-mask.h:174-176
            //            phosphor-mask-resizing.h:476-484, 626-672
            constexpr float mask_triads_per_tile = 8.0f;
            float triad_size = max(u.mask_triad_size, 1.0f);
            float tile_size  = mask_triads_per_tile * triad_size;
            float2 mask_tex_uv = (float2(gid) + 0.5f) / float2(tile_size);
            float2 ddx_uv = float2(1.0f / tile_size, 0.0f);
            float2 ddy_uv = float2(0.0f, 1.0f / tile_size);
            mask = mask_lut.sample(lut_sam, mask_tex_uv,
                                    gradient2d(ddx_uv, ddy_uv)).rgb;
        } else {
            float triad_size = max(u.mask_triad_size, 1.0f);
            float subpixel_pos = fmod(float(gid.x), triad_size) *
                                 (3.0f / triad_size);
            mask = float3(0.0f);
            if      (subpixel_pos < 1.0f) mask.r = 1.0f;
            else if (subpixel_pos < 2.0f) mask.g = 1.0f;
            else                          mask.b = 1.0f;
        }

        output.write(float4(scanline_color * mask, 1.0f), gid);
    }

} // namespace crt_royale
