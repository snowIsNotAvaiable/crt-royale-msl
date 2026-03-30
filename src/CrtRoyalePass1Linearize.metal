// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Pass 1: Linearize CRT Gamma & Bob Interlaced Fields
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration
// -----------------------------------------------------------------------------
//
// What this pass does:
//   1. Converts the input image from CRT gamma space to linear color space
//      using: linear_color = pow(input_color, crt_gamma)
//   2. If the source is interlaced, "bobs" the fields by interpolating
//      between the current scanline and its neighbors. This prevents
//      blur artifacts in subsequent passes.
//
// Input:  Original captured screen texture (gamma-encoded)
// Output: Linear-light texture, ready for scanline processing in Pass 2
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // Uniforms shared across all CRT-Royale passes.
    // This struct is mirrored in Swift (CrtRoyale.swift).
    struct Uniforms {

        // Gamma settings
        float crt_gamma;        // Simulated CRT gamma (default: 2.5)
        float lcd_gamma;        // Your display's gamma (default: 2.2)

        // Interlacing
        uint  interlace_detect; // 1 = detect and handle interlacing
        uint  interlace_bff;    // 1 = bottom field first
        uint  interlace_1080i;  // 1 = detect 1080i as interlaced
        uint  frame_count;      // Current frame number (for field alternation)

        // Source dimensions
        float2 texture_size;    // Size of the input texture in pixels
        float2 video_size;      // Size of the video content in pixels
        float2 output_size;     // Size of the output texture in pixels
    };

    // -----------------------------------------------------------------------
    // Helper: Is the source interlaced?
    // CRT-Royale considers a source interlaced if vertical resolution > 288
    // (or > 1000 when interlace_1080i is enabled, to detect 1080i content).
    // -----------------------------------------------------------------------
    inline bool is_interlaced(float video_height, uint interlace_1080i) {
        // Standard interlacing threshold from scanline-functions.h
        // A typical PAL field is 288 lines, NTSC is 240. Full-frame is double.
        float min_lines = (interlace_1080i == 1) ? 1000.0 : 288.0;
        return video_height > min_lines;
    }

    // -----------------------------------------------------------------------
    // Helper: Linearize a color value from CRT gamma space
    // This converts from the gamma-encoded CRT signal to linear light,
    // which is needed for physically correct blending in later passes.
    // -----------------------------------------------------------------------
    inline float4 linearize(float4 color, float crt_gamma) {
        return float4(pow(color.rgb, float3(crt_gamma)), color.a);
    }

    // -----------------------------------------------------------------------
    // Helper: Encode output for intermediate passes
    // For intermediate FBOs (not the final output), we don't apply output
    // gamma -- we stay in linear space. This function is a no-op for now
    // but exists to match the original shader's encode_output() pattern.
    // In later passes we may encode to sRGB or apply lcd_gamma.
    // -----------------------------------------------------------------------
    inline float4 encode_output(float4 color) {
        // Intermediate pass: output linear color as-is
        return color;
    }

    // -----------------------------------------------------------------------
    // Pass 1 Compute Kernel: Linearize & Bob
    //
    // For each output pixel:
    //   - Sample the input texture and apply CRT gamma to linearize
    //   - If interlacing is detected, blend with neighboring scanlines
    //     to "bob" (reconstruct) the missing field
    // -----------------------------------------------------------------------
    kernel void pass1_linearize(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // Gamma-encoded source
        texture2d<float, access::write>  output  [[ texture(1) ]],  // Linear-light output
        constant Uniforms               &u       [[ buffer(0)  ]],
        sampler                         sam      [[ sampler(0) ]],
        uint2                           gid      [[ thread_position_in_grid ]]
    ) {
        // Bounds check
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        // Compute normalized UV coordinates (center of pixel)
        float2 uv = (float2(gid) + 0.5) / float2(output.get_width(), output.get_height());

        // Small epsilon to avoid edge sampling artifacts (from original shader)
        uv *= 1.00001;

        // Vertical step size = one texel in the input texture
        float2 v_step = float2(0.0, 1.0 / u.texture_size.y);

        if (u.interlace_detect != 0 &&
            is_interlaced(u.video_size.y, u.interlace_1080i))
        {
            // --- Interlaced path: Bob the fields ---

            // Sample current line and neighbors, linearizing each
            float3 curr_line = linearize(input.sample(sam, uv), u.crt_gamma).rgb;
            float3 last_line = linearize(input.sample(sam, uv - v_step), u.crt_gamma).rgb;
            float3 next_line = linearize(input.sample(sam, uv + v_step), u.crt_gamma).rgb;

            // Average of the lines above and below
            float3 interpolated_line = 0.5 * (last_line + next_line);

            // Determine which field this scanline belongs to.
            // modulus = 2.0 when interlaced (alternates every other line)
            float interlaced_flag = 1.0;  // is_interlaced returned true
            float modulus = interlaced_flag + 1.0;  // = 2.0
            float field_offset = fmod(float(u.frame_count) + float(u.interlace_bff), modulus);
            float curr_line_texel = uv.y * u.texture_size.y;

            // under_half = 0.5 - a small epsilon to fix rounding at exact texel boundaries
            float under_half = 0.5 - 0.001;
            float line_num_last = floor(curr_line_texel - under_half);
            float wrong_field = fmod(line_num_last + field_offset, modulus);

            // Blend: if this line is in the "wrong" field, use the interpolated version
            float3 color = mix(curr_line, interpolated_line, wrong_field);

            output.write(encode_output(float4(color, 1.0)), gid);
        }
        else
        {
            // --- Non-interlaced path: Just linearize ---
            float4 color = linearize(input.sample(sam, uv), u.crt_gamma);
            output.write(encode_output(color), gid);
        }
    }

} // namespace crt_royale
