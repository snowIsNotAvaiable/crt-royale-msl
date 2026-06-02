// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - BLOOM_APPROX (slang pass 2)
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang reference: crt-royale-bloom-approx.{slang,h}
// Slang pass index: 2     (between vertical-scanlines and halation-V).
// Our pipeline order: linearize -> vertical-scanlines -> BLOOM_APPROX (here)
//                  -> apply-mask -> brightpass -> bloom V/H -> geometry.
//
// Output format: absolute 320x240 RGBA float, regardless of source size.
//
// Filter modes in slang (selected by `bloom_approx_filter`):
//   0    -- bilinear single-sample (1 sample/pixel)
//   1    -- 3x3 resize blur with dynamic Gaussian sigma (9 samples)
//   2    -- 4x4 true Gaussian resize (slang default, 16 samples)
//
// THIS port covers mode 2 (slang default). An earlier iteration used the
// bilinear path with a `--params bloom_approx_filter=0.0` override on the
// reference-capture side; the override is no longer needed -- delta-E is
// now measured against pure Slang defaults.
//
// What slang pass 2 does (mode 2, no PHOSPHOR_BLOOM_FAKE, default convergence
// offsets = 0 so beam_misconvergence reduces to single-channel sampling):
//   1. Compute bloom_approx_sigma from output_size, video_size, mask_triad_size
//      via the static-config get_bloom_approx_sigma path (slang ref
//      bloom-functions.h:229-290).
//   2. tex2Dresize_gaussian4x4: snap to texels for upsizing, sample 16
//      positions in a 4x4 grid around the destination pixel, compute
//      Gaussian weights from pixel-space distance, normalize.
//   3. Write the weighted sum to the 320x240 FBO (no gamma encoding).
//
// Skipped from the slang reference (TODO):
//   - 3x3 resize blur path (mode 1)
//   - beam_misconvergence with non-zero convergence offsets (would triple
//     the sample count to 48). Default offsets are zero, so disabled.
//   - PHOSPHOR_BLOOM_FAKE alternative output sizing.
//   - Slang's bug-compatible `tex2D_linearize(tex, dx)` at sample index 2
//     (bloom-approx.h:176): we replicate it exactly because librashader does.
//
// NOTE: This file is the standalone study mirror; the kernel that gets built
// lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // Sigma calculation. With slang's default chain (mask_min_allowed_triad_size
    // = 2, max_mask_texel_border = 1, mask_resize_viewport_scale = 0.0625),
    // min_allowed_viewport_triads.x evaluates to ~144. The runtime triad count
    // gets clamped to that floor for small viewports -- so the sigma's "static
    // limit" component is constant for typical setups. The viewport-dependent
    // multiplier (output.x/max_viewport_size_x) does most of the actual
    // scaling.
    inline float bloom_approx_sigma(float output_size_x,
                                    float estimated_viewport_size_x,
                                    float mask_triad_size)
    {
        constexpr float min_allowed_viewport_triads_x = 144.0f;
        constexpr float max_viewport_size_x = 1080.0f * 1024.0f * (4.0f/3.0f);
        constexpr float bloom_diff_thresh   = 1.0f / 256.0f;
        constexpr float beam_max_sigma      = 0.3f;

        float num_triads_from_size = estimated_viewport_size_x /
                                     max(mask_triad_size, 1e-5f);
        float mask_num_triads_runtime =
            max(min_allowed_viewport_triads_x, num_triads_from_size);
        float asymptotic_triad_size = max_viewport_size_x / mask_num_triads_runtime;
        float asymptotic_sigma = -0.05168f + 0.6113f * asymptotic_triad_size -
            1.122f * asymptotic_triad_size * sqrt(0.000416f + bloom_diff_thresh);
        float sigma_x = asymptotic_sigma * output_size_x / max_viewport_size_x;
        return length(float2(sigma_x, beam_max_sigma));
    }

    // 16-sample true Gaussian resize. Snaps to texels when upsizing, computes
    // per-sample Gaussian weight from squared pixel-space distance to dest.
    inline float3 tex2Dresize_gaussian4x4(
        texture2d<float, access::sample> tex, sampler sam,
        float2 tex_uv, float2 dxdy, float2 tex_size, float2 tex_size_inv,
        float2 tex_uv_to_pixel_scale, float sigma)
    {
        constexpr float UNDER_HALF = 0.4995f;
        float denom_inv = 0.5f / (sigma * sigma);

        float2 curr_texel = tex_uv * tex_size;
        float2 prev_texel = floor(curr_texel - float2(UNDER_HALF)) + float2(0.5f);
        float2 prev_texel_uv = prev_texel * tex_size_inv;
        float2 snap = float2(dxdy.x <= tex_size_inv.x ? 1.0f : 0.0f,
                             dxdy.y <= tex_size_inv.y ? 1.0f : 0.0f);
        float2 sample5_uv = mix(tex_uv - 0.5f * dxdy, prev_texel_uv, snap);

        float2 dx = float2(dxdy.x, 0.0f);
        // Build the 4x4 sample grid (slang index numbering preserved).
        // ... 16 UV positions, 16 samples, 16 weights, normalize ...
        // (Full code lives in the production CrtRoyale.metal; the structure
        // matches slang bloom-approx.h:148-233 exactly.)

        // Returned value: weighted-average of the 16 samples.
        return tex.sample(sam, sample5_uv).rgb; // placeholder for the doc mirror
    }

    kernel void pass_bloom_approx(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 src_size = float2(input.get_width(),  input.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;
        float2 dxdy_min_scale = src_size / out_size;
        float2 dxdy_scale = max(dxdy_min_scale, float2(1.0f));
        float2 tex_size_inv = 1.0f / src_size;
        float2 dxdy = dxdy_scale * tex_size_inv;
        constexpr float geom_aspect_ratio_static = 1.313069909f;
        float estimated_viewport_size_x = u.video_size.y * geom_aspect_ratio_static;
        float sigma = bloom_approx_sigma(out_size.x, estimated_viewport_size_x,
                                         u.mask_triad_size);
        float3 color = tex2Dresize_gaussian4x4(input, sam, uv, dxdy, src_size,
                                               tex_size_inv, out_size, sigma);
        output.write(float4(color, 1.0f), gid);
    }

} // namespace crt_royale
