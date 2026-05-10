// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Pass 3: Apply Phosphor Mask (simplified)
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
// Static-config simplifications relative to the slang reference:
//   - No horizontal beam-filter shaping (sample_rgb_scanline_horizontal):
//     direct linear sample of the input. Equivalent to beam_horiz_filter
//     defaulting to a delta function rather than Quilez/Lanczos2.
//   - convergence_offset_x_{r,g,b} = 0 (no chromatic aberration).
//   - halation_weight = 0 (this is the slang default too, but explicit here).
//   - No PHOSPHOR_BLOOM_FAKE.
//   - mask_type forced to 0 (aperture grille). Slot/shadow masks need the LUT
//     loader, planned for a future iteration.
//   - Mask is generated procedurally (not loaded from a PNG LUT).
//
// Future-work hooks (commented in the kernel body):
//   - LUT-based mask sampling for slot/shadow masks
//   - Quilez / Lanczos2 horizontal beam shaping
//   - Halation desaturation when HALATION_BLUR is available
//   - Bloom-fake blend when BLOOM_APPROX is available
//
// NOTE: This file is the standalone study mirror; the kernel that gets built
// lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // The full Uniforms struct lives in CrtRoyale.metal. Relevant fields here:
    //
    //   uint  mask_type;            // 0 = aperture grille (only one supported)
    //   float mask_triad_size;      // output pixels per RGB triad
    //   float mask_amplify;         // brightness compensation factor

    kernel void pass3_apply_mask(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());
        float3 scanline_color = input.sample(sam, uv).rgb;

        // Aperture grille: position in [0, 3) within the current triad.
        float triad_size = max(u.mask_triad_size, 1.0f);
        float subpixel_pos = fmod(float(gid.x), triad_size) * (3.0f / triad_size);

        float3 mask = float3(0.0f);
        if      (subpixel_pos < 1.0f) mask.r = 1.0f;
        else if (subpixel_pos < 2.0f) mask.g = 1.0f;
        else                          mask.b = 1.0f;

        float3 phosphor = scanline_color * mask * u.mask_amplify;
        output.write(float4(phosphor, 1.0f), gid);
    }

} // namespace crt_royale
