// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Pass 2: Vertical Scanlines / Beam Distribution
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration
// -----------------------------------------------------------------------------
//
// What this pass does:
//   For each output pixel, sample the bracketing scanlines (and one outside
//   neighbour) and weight each with a generalized-Gaussian beam profile whose
//   width (sigma) and shape (beta) vary with the source scanline brightness.
//   The result is the integrated light arriving at the pixel from all visible
//   scanlines -- the characteristic CRT scanline look.
//
// Input:  Linear-light texture from Pass 1 (already gamma-linearized).
//         IMPORTANT: do NOT re-apply crt_gamma here. The reference shader's
//         tex2D_linearize() collapses to a plain texture sample once we
//         standardize on linear-light intermediates.
//
// Output: Linear-light texture with vertical scanline shaping applied. Auto-
//         dimmed by levels_autodim_temp = 0.5 to leave clipping headroom.
//
// Static-config first cut (matches user-settings.h defaults):
//   beam_num_scanlines        = 3      (the > 2.5 branch in the slang ref)
//   beam_generalized_gaussian = true
//   beam_antialias_level      = 1.0    (sampled, 3-tap average)
//   beam_misconvergence       = false
//   beam_spot_shape_function  = 0.0    (power form for sigma)
// Dropping these branches halves the kernel code and matches the most-used
// configuration. Kann-Kriterien (variable scanline count, true integral AA,
// pure-Gaussian fallback, RGB convergence) extend this kernel later.
//
// NOTE: This file is the standalone study mirror that tracks the pass in
// isolation. The actually-built kernel lives in
// vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal alongside Pass 1.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // The full Uniforms struct lives in CrtRoyale.metal -- declared here as a
    // study stub so this file documents the pass's contract.
    //
    // struct Uniforms {
    //     float crt_gamma, lcd_gamma;
    //     uint  interlace_detect, interlace_bff, interlace_1080i, frame_count;
    //     float2 texture_size, video_size, output_size;
    //     float beam_min_sigma, beam_max_sigma, beam_spot_power;
    //     float beam_min_shape, beam_max_shape, beam_shape_power;
    //     float levels_autodim_temp;
    //     uint  debug_pass_index;
    // };

    // Slang ref: derived-settings-and-constants.h:311.
    constant float UNDER_HALF = 0.4995f;

    // Slang ref: scanline-functions.h:540-567 (SD interlace 288.5..576.5,
    // optional 1080i band).
    inline bool is_interlaced(float num_lines, uint interlace_1080i) {
        bool sd = (num_lines > 288.5f) && (num_lines < 576.5f);
        bool hd = (interlace_1080i == 1) &&
                  (num_lines > 1079.5f) && (num_lines < 1080.5f);
        return sd || hd;
    }

    // Lanczos approximation of the Gamma function. Slang ref:
    // include/special-functions.h:200-211. Caller must precompute s_inv = 1/s.
    inline float3 gamma_impl(float3 s, float3 s_inv) {
        constexpr float g  = 1.12906830989f;
        constexpr float c0 = 0.8109119309638332633713423362694399653724431f;
        constexpr float c1 = 0.4808354605142681877121661197951496120000040f;
        constexpr float e  = 2.71828182845904523536028747135266249775724709f;
        float3 sph         = s + 0.5f;
        float3 lanczos_sum = float3(c0) + float3(c1) / (s + 1.0f);
        float3 base        = (sph + g) / e;
        return (pow(base, sph) * lanczos_sum) * s_inv;
    }

    // sigma(color) = beam_min_sigma + sigma_range * color^beam_spot_power.
    // Slang ref: scanline-functions.h:33-95 (power-shape branch).
    inline float3 get_gaussian_sigma(float3 color, float sigma_range,
                                     float beam_min_sigma, float beam_spot_power)
    {
        return float3(beam_min_sigma) +
               sigma_range * pow(color, float3(beam_spot_power));
    }

    // beta(color) = beam_min_shape + shape_range * color^beam_shape_power.
    // Slang ref: scanline-functions.h:97-124.
    inline float3 get_generalized_gaussian_beta(float3 color, float shape_range,
                                                float beam_min_shape,
                                                float beam_shape_power)
    {
        return float3(beam_min_shape) +
               shape_range * pow(color, float3(beam_shape_power));
    }

    // 3-sample-averaged generalized-Gaussian beam contribution.
    // Slang ref: scanline-functions.h:225-254 (the if-branch since default
    // beam_antialias_level = 1.0 > 0.5).
    inline float3 scanline_generalized_gaussian_sampled_contrib(
        float3 dist, float3 color, float pixel_height,
        float sigma_range, float shape_range,
        float beam_min_sigma, float beam_spot_power,
        float beam_min_shape, float beam_shape_power)
    {
        float3 sigma = get_gaussian_sigma(color, sigma_range,
                                          beam_min_sigma, beam_spot_power);
        float3 alpha = sqrt(2.0f) * sigma;
        float3 beta  = get_generalized_gaussian_beta(color, shape_range,
                                                    beam_min_shape, beam_shape_power);
        float3 alpha_inv = 1.0f / alpha;
        float3 beta_inv  = 1.0f / beta;

        float3 scale = color * beta * 0.5f * alpha_inv /
                       gamma_impl(beta_inv, beta);

        float3 sample_offset = float3(pixel_height / 3.0f);
        float3 dist2 = dist + sample_offset;
        float3 dist3 = abs(dist - sample_offset);

        float3 w1 = exp(-pow(abs(dist  * alpha_inv), beta));
        float3 w2 = exp(-pow(abs(dist2 * alpha_inv), beta));
        float3 w3 = exp(-pow(abs(dist3 * alpha_inv), beta));

        return scale / 3.0f * (w1 + w2 + w3);
    }

    // Slang ref: scanline-functions.h:510-538.
    inline float2 get_last_scanline_uv(float2 tex_uv, float2 tex_size,
                                       float2 tex_size_inv,
                                       float2 il_step_multiple,
                                       uint frame_count, uint interlace_bff,
                                       thread float& dist)
    {
        float field_offset = floor(il_step_multiple.y * 0.75f) *
                             fmod(float(frame_count) + float(interlace_bff), 2.0f);
        float2 curr_texel     = tex_uv * tex_size;
        float2 prev_texel_num = floor(curr_texel - float2(UNDER_HALF));
        float wrong_field     = fmod(prev_texel_num.y + field_offset,
                                     il_step_multiple.y);
        float2 scanline_texel_num = prev_texel_num - float2(0.0f, wrong_field);
        float2 scanline_texel = scanline_texel_num + float2(0.5f);
        float2 scanline_uv    = scanline_texel * tex_size_inv;
        dist = (curr_texel.y - scanline_texel.y) / il_step_multiple.y;
        return scanline_uv;
    }

} // namespace crt_royale

// Kernel body (study mirror; see CrtRoyale.metal for the actual build target)
//
// kernel void pass2_vertical_scanlines(
//     texture2d<float, access::sample> input,
//     texture2d<float, access::write>  output,
//     constant Uniforms& u,
//     sampler sam,
//     uint2 gid)
// {
//     ... see vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal
// }
