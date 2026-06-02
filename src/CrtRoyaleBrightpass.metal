// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Brightpass (slang pass 8)
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang reference: crt-royale-brightpass.{slang,h}
// Slang pass index: 8
//
// Inputs (texture indices):
//   0  MASKED_SCANLINES  -- our pass 3 output (slang's pass 7 alias)
//   2  BLOOM_APPROX      -- our pass_bloom_approx output (slang's pass 2)
// Output:
//   1  BRIGHTPASS        -- consumed by bloom V/H (slang passes 9, 10)
//
// What this pass does:
//   For each pixel we compute "how much extra blur energy will arrive from
//   neighboring blooming phosphors", expressed as a per-channel blur_ratio in
//   [0, 1]. The output is the masked-scanline value multiplied by the ratio:
//   bright pixels stop here (they bloom into their neighbors), dim pixels
//   pass through (their neighbors will fill them in via bloom).
//
//   The derivation lives in TroggleMonkey's original comments in
//   crt-royale-brightpass.h:113-163; we reproduce the algebra 1:1.
//
// Static-config simplifications:
//   - bloom_sigma + center_weight computed from `mask_triad_size` uniform
//     (matches slang's `mask_triad_size_desired_static` default 3.0).
//   - Default 9-tap blur center weight only. PHOSPHOR_BLOOM_TRIADS_LARGER_*
//     branches not implemented (TODO when needed).
//   - levels_contrast = 1.0 (slang default).
//   - bloom_underestimate_levels = 0.8 (slang default, hardcoded).
//   - bloom_excess = 0 (slang default, hardcoded).
//   - BRIGHTPASS_AREA_BASED branch skipped (slang default).
//
// NOTE: This file is the standalone study mirror; the kernel that gets built
// lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // Slang's get_min_sigma_to_blur_triad (bloom-functions.h:46). Returns the
    // smallest Gaussian sigma that fully blurs a triad of the given pixel
    // size to within `thresh` per pixel.
    inline float bp_get_min_sigma_to_blur_triad(float triad_size, float thresh) {
        return -0.05168f + 0.6113f * triad_size -
               1.122f * triad_size * sqrt(0.000416f + thresh);
    }

    // Slang's get_center_weight for the 9-tap default blur path. The full
    // function in bloom-functions.h:95-160 branches on PHOSPHOR_BLOOM_TRIADS_
    // LARGER_THAN_{3,6,9,12}_PIXELS for 17/25/31/43-tap variants; we cover
    // only the 9-tap fallback here.
    inline float bp_get_center_weight_9tap(float sigma) {
        float denom_inv = 0.5f / (sigma * sigma);
        float w0 = 1.0f;
        float w1 = exp(-1.0f  * denom_inv);
        float w2 = exp(-4.0f  * denom_inv);
        float w3 = exp(-9.0f  * denom_inv);
        float w4 = exp(-16.0f * denom_inv);
        float weight_sum_inv = 1.0f / (w0 + 2.0f * (w1 + w2 + w3 + w4));
        return weight_sum_inv * weight_sum_inv;
    }

    kernel void pass_brightpass(
        texture2d<float, access::sample> masked_scanlines [[ texture(0) ]],
        texture2d<float, access::write>  output           [[ texture(1) ]],
        texture2d<float, access::sample> bloom_approx     [[ texture(2) ]],
        constant Uniforms&               u                [[ buffer(0)  ]],
        sampler                          sam              [[ sampler(0) ]],
        uint2                            gid              [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());

        float3 intensity_dim = masked_scanlines.sample(sam, uv).rgb;
        float3 phosphor_blur_approx = bloom_approx.sample(sam, uv).rgb;

        // intensity = full mask-amplified version of MASKED_SCANLINES, as if
        // we had undimmed the mask in pass 3. Slang ref:
        //   intensity = intensity_dim * undim_factor * mask_amplify * levels_contrast
        // levels_contrast = 1.0 (default, not exposed).
        float undim_factor = 1.0f / max(u.levels_autodim_temp, 1e-5f);
        float3 intensity = intensity_dim * undim_factor * u.mask_amplify;

        constexpr float bloom_diff_thresh = 1.0f / 256.0f;
        float bloom_sigma = bp_get_min_sigma_to_blur_triad(
            u.mask_triad_size, bloom_diff_thresh);
        float center_weight = bp_get_center_weight_9tap(bloom_sigma);

        float3 max_area_contribution_approx =
            max(float3(0.0f), phosphor_blur_approx - center_weight * intensity);

        constexpr float bloom_underestimate_levels = 0.8f;
        float3 area_contrib_underestimate =
            bloom_underestimate_levels * max_area_contribution_approx;
        float3 intensity_underestimate =
            bloom_underestimate_levels * intensity;

        // blur_ratio_temp = ((1 - area) / intensity - 1) / (center_weight - 1)
        // Saturate intensity to avoid div-by-zero on solid black.
        float3 safe_intensity = max(intensity_underestimate, float3(1e-5f));
        float3 blur_ratio_temp =
            ((float3(1.0f) - area_contrib_underestimate) / safe_intensity -
             float3(1.0f)) / (center_weight - 1.0f);
        float3 blur_ratio = clamp(blur_ratio_temp, float3(0.0f), float3(1.0f));

        constexpr float bloom_excess = 0.0f;
        float3 brightpass = intensity_dim *
            mix(blur_ratio, float3(1.0f), bloom_excess);

        output.write(float4(brightpass, 1.0f), gid);
    }

} // namespace crt_royale
