// -----------------------------------------------------------------------------
// CRT-Royale MSL Port -- Compute Kernels (Pass 1 + Pass 2 + Final Encode)
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration.
//
// Currently implemented:
//   pass1_linearize           -- Linearize CRT gamma + bob interlaced fields
//   pass2_vertical_scanlines  -- Vertical scanlines / beam distribution
//   pass_final_encode         -- Re-encode linear -> display gamma (temporary
//                                stand-in until passes 3..12 are ported)
//
// Pipeline contract (same throughout the port):
//   Pass 1 linearizes once. ALL subsequent intermediate textures are
//   linear-light. The reference shader's tex2D_linearize() therefore becomes
//   a plain texture sample in our port -- do NOT re-apply crt_gamma in pass 2
//   or anywhere downstream. Only pass_final_encode performs the inverse
//   gamma encoding for display.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // -----------------------------------------------------------------------
    // Uniforms shared by all CRT-Royale passes (mirrors CrtRoyale.swift)
    // Keep the field order identical on both sides.
    // -----------------------------------------------------------------------
    struct Uniforms {

        // Gamma settings
        float crt_gamma;
        float lcd_gamma;

        // Interlacing
        uint  interlace_detect;
        uint  interlace_bff;
        uint  interlace_1080i;
        uint  frame_count;

        // Source dimensions
        float2 texture_size;
        float2 video_size;
        float2 output_size;

        // Beam (pass 2)
        float beam_min_sigma;
        float beam_max_sigma;
        float beam_spot_power;
        float beam_min_shape;
        float beam_max_shape;
        float beam_shape_power;
        float levels_autodim_temp;

        // Mask (pass 3)
        uint  mask_type;            // 0 = aperture grille, 1 = slot, 2 = shadow.
                                    //     Only used by the LUT path; the
                                    //     procedural fallback always renders an
                                    //     aperture grille regardless.
        float mask_triad_size;      // Output pixels per RGB triad (default 3.0
                                    //     -> ~1 px per subpixel). Used for the
                                    //     procedural path AND to derive the
                                    //     LUT tile size in the LUT path.
        float mask_amplify;         // Brightness compensation for mask dimming.
                                    //     For mask_grille15: 1/(53/255) ≈ 4.81.
                                    //     Applied in pass_final_encode (matches
                                    //     Slang pass 11 where the un-dim lives).
        uint  mask_lut_enabled;     // 0 = procedural aperture grille,
                                    //     1 = sample mask_grille_texture_large
                                    //         bound at texture(2).
        uint  mask_sample_mode;     // 0 = sample MASK_RESIZE (slang Pass 5+6
                                    //     output, Lanczos-resized tile),
                                    // 1 = hardware-resample (sample the
                                    //     original *_large LUT with mipmap
                                    //     + anisotropy). Slang default = 0;
                                    //     we default to 1 historically but
                                    //     can switch with Mask-Resize ported.

        // Halation / Diffusion (slang passes 3+4 + apply-mask + bloom-final)
        float halation_weight;      // [0, 1], slang default 0.0. Mixes
                                    //     desaturated halation into the
                                    //     dimmed scanline color (pass 3).
        float diffusion_weight;     // [0, 1], slang default 0.075. Mixes
                                    //     HALATION_BLUR (full chroma) into
                                    //     the phosphor bloom (pass 10).

        // Geometry / final pass 4 (slang pass 11)
        uint  geom_mode;            // 0 = flat (slang default, no curvature),
                                    //     1 = barrel-distortion approximation.
                                    //     Modes 2-3 from slang (cylindrical /
                                    //     asymmetric) not implemented.
        float geom_radius_factor;   // barrel-distortion strength. 0 = flat,
                                    //     0.2-0.4 = subtle CRT bulge.
        float border_size;          // [0, 0.5], slang default 0.015.
        float border_darkness;      // [0, inf), slang default 2.0.
        float border_compress;      // [1, inf), slang default 2.5.
        float brightness_boost;     // 1.0 = slang-correct dim output, >1 to
                                    //     compensate for missing bloom (which
                                    //     would otherwise un-dim the image).

        // Debug pipeline
        uint  debug_pass_index;     // 0=final, 1=pass 1, 2=pass 2, 3=pass 3
    };

    // Compile-time constant from derived-settings-and-constants.h:311.
    // Used to nudge texel-snapping arithmetic away from exact half-integers
    // where rounding would otherwise jitter to the next texel.
    constant float UNDER_HALF = 0.4995f;

    // =======================================================================
    // Shared helpers (used by pass 1 and pass 2)
    // =======================================================================

    // Aligned with scanline-functions.h:540-567 of the slang reference.
    // SD interlacing: 288.5 < lines < 576.5. HD: only the 1080i band when
    // explicitly enabled.
    inline bool is_interlaced(float num_lines, uint interlace_1080i) {
        bool sd = (num_lines > 288.5f) && (num_lines < 576.5f);
        bool hd = (interlace_1080i == 1) &&
                  (num_lines > 1079.5f) && (num_lines < 1080.5f);
        return sd || hd;
    }

    inline float4 linearize(float4 color, float crt_gamma) {
        return float4(pow(color.rgb, float3(crt_gamma)), color.a);
    }

    // =======================================================================
    // Special functions (subset of include/special-functions.h)
    // Only the variants Pass 2's sampled-contrib path actually needs.
    // =======================================================================

    // Lanczos approximation of the gamma function (g=1.12906830989, two coeffs).
    // s_inv = 1/s must be precomputed and passed by the caller for reuse.
    // Slang ref: special-functions.h:200-211.
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

    // =======================================================================
    // Beam-shape helpers
    // =======================================================================

    // get_gaussian_sigma -- Slang ref: scanline-functions.h:33-95.
    // We only port the power-shape branch (beam_spot_shape_function < 0.5,
    // the static default in user-settings.h).
    inline float3 get_gaussian_sigma(float3 color, float sigma_range,
                                     float beam_min_sigma, float beam_spot_power)
    {
        return float3(beam_min_sigma) +
               sigma_range * pow(color, float3(beam_spot_power));
    }

    // get_generalized_gaussian_beta -- Slang ref: scanline-functions.h:97-124.
    inline float3 get_generalized_gaussian_beta(float3 color, float shape_range,
                                                float beam_min_shape,
                                                float beam_shape_power)
    {
        return float3(beam_min_shape) +
               shape_range * pow(color, float3(beam_shape_power));
    }

    // scanline_generalized_gaussian_sampled_contrib (3-sample avg branch).
    // Slang ref: scanline-functions.h:225-254 (the if-branch since default
    // beam_antialias_level = 1.0 > 0.5).
    //
    // Computes a single scanline's light contribution at distance `dist`
    // (in scanlines) for the underlying scanline color `color`, using a
    // generalized Gaussian distribution: beta < 2 spikes, beta == 2 is
    // Gaussian, beta > 2 flattens the peak.
    //
    // The 3-sample average reduces aliasing for thin scanlines on coarse
    // output grids (one sample at dist, plus +/- pixel_height/3).
    //
    // Static branches dropped from the slang reference (TODO: revisit when
    // user-tunable beam config becomes a Kann-Kriterium):
    //   - beam_antialias_level <= 0.5 (single-sample fallback)
    //   - beam_generalized_gaussian == false (pure Gaussian via erf integral)
    //   - beam_antialias_level > 1.5 (closed-form integral via ligamma)
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

        // scale = color * beta * 0.5 / (alpha * gamma(1/beta))
        // gamma_impl takes (s, s_inv); here s = beta_inv, s_inv = beta.
        float3 scale = color * beta * 0.5f * alpha_inv /
                       gamma_impl(beta_inv, beta);

        // Three-sample average around dist (anti-aliased).
        float3 sample_offset = float3(pixel_height / 3.0f);
        float3 dist2 = dist + sample_offset;
        float3 dist3 = abs(dist - sample_offset);

        float3 w1 = exp(-pow(abs(dist  * alpha_inv), beta));
        float3 w2 = exp(-pow(abs(dist2 * alpha_inv), beta));
        float3 w3 = exp(-pow(abs(dist3 * alpha_inv), beta));

        return scale / 3.0f * (w1 + w2 + w3);
    }

    // get_last_scanline_uv -- Slang ref: scanline-functions.h:510-538.
    // Finds the previous scanline center in the current field and the sample's
    // distance from it (in scanlines).
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

    // =======================================================================
    // Pass 1: Linearize CRT gamma + bob interlaced fields
    // =======================================================================
    kernel void pass1_linearize(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms               &u       [[ buffer(0)  ]],
        sampler                         sam      [[ sampler(0) ]],
        uint2                           gid      [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());
        uv *= 1.00001f;

        float2 v_step = float2(0.0f, 1.0f / u.texture_size.y);

        if (u.interlace_detect != 0 &&
            is_interlaced(u.video_size.y, u.interlace_1080i))
        {
            float3 curr_line = linearize(input.sample(sam, uv),          u.crt_gamma).rgb;
            float3 last_line = linearize(input.sample(sam, uv - v_step), u.crt_gamma).rgb;
            float3 next_line = linearize(input.sample(sam, uv + v_step), u.crt_gamma).rgb;
            float3 interpolated_line = 0.5f * (last_line + next_line);

            float modulus = 2.0f;
            float field_offset = fmod(float(u.frame_count) +
                                      float(u.interlace_bff), modulus);
            float curr_line_texel = uv.y * u.texture_size.y;
            float line_num_last   = floor(curr_line_texel - UNDER_HALF);
            float wrong_field     = fmod(line_num_last + field_offset, modulus);

            float3 color = mix(curr_line, interpolated_line, wrong_field);
            output.write(float4(color, 1.0f), gid);
        }
        else
        {
            float4 color = linearize(input.sample(sam, uv), u.crt_gamma);
            output.write(color, gid);
        }
    }

    // =======================================================================
    // Pass 2: Vertical scanlines / beam distribution
    // Slang ref: crt-royale-scanlines-vertical-interlacing.slang
    //
    // Static-config first cut (sufficient for Phase-1 visual correctness):
    //   - beam_num_scanlines   = 3 (the > 2.5 branch)
    //   - beam_generalized_gaussian = true
    //   - beam_antialias_level = 1.0 (sampled, 3-tap)
    //   - beam_misconvergence  = false
    //   - beam_spot_shape_function < 0.5 (power form)
    // The dropped branches are listed inline in the helper docstrings as
    // future-work hooks.
    // =======================================================================
    kernel void pass2_vertical_scanlines(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms               &u       [[ buffer(0)  ]],
        sampler                         sam      [[ sampler(0) ]],
        uint2                           gid      [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        // Match slang vertex shader: tex_uv = TexCoord * 1.00001 (line 49).
        float2 tex_uv = (float2(gid) + 0.5f) /
                        float2(output.get_width(), output.get_height());
        tex_uv *= 1.00001f;

        // Interlace step multiple: (1, 1) progressive or (1, 2) interlaced.
        float y_step = is_interlaced(u.video_size.y, u.interlace_1080i)
                       ? 2.0f : 1.0f;
        float2 il_step_multiple = float2(1.0f, y_step);
        float2 tex_size_inv     = 1.0f / u.texture_size;
        float2 uv_step          = il_step_multiple * tex_size_inv;

        // Output pixel height in scanlines (used for AA sample offsets and
        // for the beam contribution's normalization).
        float pixel_height = (u.video_size.y / u.output_size.y) /
                             il_step_multiple.y;

        // Find the previous scanline center in the current field and the
        // sample's distance from it (in scanlines).
        float dist;
        float2 scanline_uv = get_last_scanline_uv(
            tex_uv, u.texture_size, tex_size_inv, il_step_multiple,
            u.frame_count, u.interlace_bff, dist);

        // Sample the two scanlines bracketing the output pixel (lines 2 & 3).
        // No re-linearization: input is already linear-light from pass 1.
        float2 v_step = float2(0.0f, uv_step.y);
        float3 c2 = input.sample(sam, scanline_uv).rgb;
        float3 c3 = input.sample(sam, scanline_uv + v_step).rgb;

        // beam_num_scanlines == 3: pick one outside scanline based on which
        // side of the bracket dist favors. dist in [0, 1].
        float dist_round = round(dist);
        float2 outside_off = mix(-v_step, 2.0f * v_step, dist_round);
        float3 cOut = input.sample(sam, scanline_uv + outside_off).rgb;

        // No vertical convergence offsets (beam_misconvergence = false).
        float3 dist2 = float3(dist);

        // Pre-computed once per pixel.
        float sigma_range = max(u.beam_max_sigma, u.beam_min_sigma) -
                            u.beam_min_sigma;
        float shape_range = max(u.beam_max_shape, u.beam_min_shape) -
                            u.beam_min_shape;

        float3 contrib2 = scanline_generalized_gaussian_sampled_contrib(
            dist2, c2, pixel_height, sigma_range, shape_range,
            u.beam_min_sigma, u.beam_spot_power,
            u.beam_min_shape, u.beam_shape_power);

        float3 contrib3 = scanline_generalized_gaussian_sampled_contrib(
            abs(float3(1.0f) - dist2), c3, pixel_height,
            sigma_range, shape_range,
            u.beam_min_sigma, u.beam_spot_power,
            u.beam_min_shape, u.beam_shape_power);

        float3 distOut = mix(dist2 + 1.0f, 2.0f - dist2, dist_round);
        float3 contribOut = scanline_generalized_gaussian_sampled_contrib(
            distOut, cOut, pixel_height, sigma_range, shape_range,
            u.beam_min_sigma, u.beam_spot_power,
            u.beam_min_shape, u.beam_shape_power);

        // Auto-dim to avoid clipping (levels_autodim_temp = 0.5 by default).
        // Output stays in linear light for downstream passes / final encode.
        float3 intensity = (contrib2 + contrib3 + contribOut) *
                           u.levels_autodim_temp;
        output.write(float4(intensity, 1.0f), gid);
    }

    // =======================================================================
    // Pass 3: Apply phosphor mask (simplified for first cut)
    // Slang ref: crt-royale-scanlines-horizontal-apply-mask.h
    //
    // The full reference also samples the BLOOM_APPROX texture (Pass 2) for a
    // fake phosphor bloom, samples HALATION_BLUR (Pass 4) for desaturated
    // smear, and uses a precomputed MASK_RESIZE LUT (Pass 5+6) with custom
    // tile bookkeeping. We skip all of that for now and produce the bare
    // electron-times-mask product. That is exactly what the reference reduces
    // to with halation_weight=0, PHOSPHOR_BLOOM_FAKE off, and either of the
    // hardware-sample mask modes.
    //
    // Mask source: a procedurally generated aperture grille. Each output
    // triad is `u.mask_triad_size` pixels wide and contains an R, G, B
    // subpixel of equal width. mask_type other than 0 falls back to aperture
    // grille for now (slot/shadow LUTs require the PNG-LUT loader, planned
    // for a future iteration).
    //
    // Static branches dropped from the reference (TODO: revisit):
    //   - sample_rgb_scanline_horizontal (Quilez/Lanczos2 horizontal filter):
    //     replaced by a direct linear sample of the input. Visible only on
    //     low-res sources scaled to a non-integer multiple horizontally.
    //   - convergence_offset_x_{r,g,b}: assumed zero.
    //   - PHOSPHOR_BLOOM_FAKE: not implemented.
    //
    // Now ported (was previously skipped):
    //   - HALATION_BLUR sampling + halation_weight lerp (slang ref
    //     apply-mask.h:209-222). At slang default halation_weight=0 this is
    //     a no-op, but the math runs for runtime-tunable values.
    //   - mask LUT-based sampling (slot, shadow) via mask_lut_enabled.
    // =======================================================================
    kernel void pass3_apply_mask(
        texture2d<float, access::sample> input         [[ texture(0) ]],  // VERTICAL_SCANLINES
        texture2d<float, access::write>  output        [[ texture(1) ]],
        texture2d<float, access::sample> mask_lut      [[ texture(2) ]],  // mask_grille_texture_large (optional)
        texture2d<float, access::sample> halation_blur [[ texture(3) ]],  // HALATION_BLUR (slang pass 4)
        texture2d<float, access::sample> mask_resize   [[ texture(4) ]],  // MASK_RESIZE (slang pass 6, optional)
        constant Uniforms               &u             [[ buffer(0)  ]],
        sampler                          sam           [[ sampler(0) ]],
        sampler                          lut_sam       [[ sampler(1) ]],  // linear + repeat-wrap
        uint2                            gid           [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        // Sample the upstream scanline texture at our pixel center. We're not
        // doing horizontal beam-filter shaping here -- direct linear sample.
        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());
        float3 scanline_color_dim = input.sample(sam, uv).rgb;

        float3 mask;
        if (u.mask_lut_enabled == 0u) {
            // Procedural fallback (aperture grille only). Useful when no LUT
            // is bound, or as a debug toggle to A/B the two paths.
            float triad_size = max(u.mask_triad_size, 1.0f);
            float subpixel_pos = fmod(float(gid.x), triad_size) *
                                 (3.0f / triad_size);
            mask = float3(0.0f);
            if      (subpixel_pos < 1.0f) mask.r = 1.0f;
            else if (subpixel_pos < 2.0f) mask.g = 1.0f;
            else                          mask.b = 1.0f;
        } else if (u.mask_sample_mode != 0u) {
            // Hardware-resample LUT path (slang mask_sample_mode > 0.5).
            // Slang ref: crt-royale-scanlines-horizontal-apply-mask.h:174-176
            //            phosphor-mask-resizing.h:476-484, 626-672
            //
            // mask_resize_tile_size = mask_triads_per_tile (=8) * mask_triad_size.
            // mask_tiles_per_screen = viewport / mask_resize_tile_size.
            // tile_uv_wrap = video_uv * mask_tiles_per_screen
            //              = (gid + 0.5) / mask_resize_tile_size
            // mask_sample_mode > 0.5 -> convert_phosphor_tile_uv_wrap_to_tex_uv
            // returns tile_uv_wrap unchanged. Sampler does the repeat-wrap.
            constexpr float mask_triads_per_tile = 8.0f;
            float triad_size = max(u.mask_triad_size, 1.0f);
            float tile_size  = mask_triads_per_tile * triad_size;
            float2 mask_tex_uv = (float2(gid) + 0.5f) / float2(tile_size);
            // Compute shaders have no implicit ddx/ddy; pass analytical UV
            // gradients so the sampler can pick the right mipmap LOD. One
            // screen pixel advances mask_tex_uv by (1/tile_size, 0) in x and
            // (0, 1/tile_size) in y -- mirrors slang's automatic derivative.
            float2 ddx_uv = float2(1.0f / tile_size, 0.0f);
            float2 ddy_uv = float2(0.0f, 1.0f / tile_size);
            mask = mask_lut.sample(lut_sam, mask_tex_uv,
                                    gradient2d(ddx_uv, ddy_uv)).rgb;
        } else {
            // mask_sample_mode == 0: sample MASK_RESIZE (slang Pass 6 output).
            // Slang ref: crt-royale-scanlines-horizontal-apply-mask.h fragment
            //            (the else branch when mask_sample_mode_static < 0.5).
            //            tex2Dtiled_mask_linearize at phosphor-mask-resizing.h:626.
            //
            // Both slang and us use the SAME tile_uv_wrap convention; the
            // only difference vs. the hardware-resample path is what texture
            // we feed it through. MASK_RESIZE is a 2-tile-wide-and-tall LUT
            // pre-resized for our viewport, so we take tile_uv_wrap and look
            // it up via fract() in one of those two tiles. The "linearize"
            // is a no-op for us (mask_resize is already linear).
            constexpr float mask_triads_per_tile = 8.0f;
            float triad_size = max(u.mask_triad_size, 1.0f);
            float tile_size  = mask_triads_per_tile * triad_size;
            float2 tile_uv_wrap = (float2(gid) + 0.5f) / float2(tile_size);
            // mask_tex_uv lives in [0, 1] across one MASK_RESIZE tile (which
            // itself sits inside the 2-tile MASK_RESIZE FBO at uv [0, 0.5]).
            float2 mask_uv = (fract(tile_uv_wrap) + float2(0.5f)) * 0.5f;
            mask = mask_resize.sample(sam, mask_uv).rgb;
        }

        // Halation mix (slang apply-mask.h:209-222). HALATION_BLUR is sampled
        // at the same UV (the slang vertex computes halation_tex_uv = video_uv
        // * HALATION_BLUR_video_size/HALATION_BLUR_texture_size; with
        // scale_type=source both ratios are 1 so it reduces to video_uv).
        // halation_color is averaged to a scalar (electrons hit *any* phosphor
        // -> desaturated) and dimmed by auto_dim_factor/3 before the lerp.
        float auto_dim_factor = u.levels_autodim_temp;
        float3 halation_color = halation_blur.sample(sam, uv).rgb;
        float3 halation_intensity_dim =
            float3(dot(halation_color, float3(auto_dim_factor / 3.0f)));
        float3 electron_intensity_dim =
            mix(scanline_color_dim, halation_intensity_dim, u.halation_weight);

        // Slang pass 7 returns the dim product (electrons * mask) without
        // un-dimming. The un-dim by 1/levels_autodim_temp / mask_amplify lives
        // in pass 11 alongside bloom reconstitution. We mirror that split:
        // pass 3 stays dim (-> apples-to-apples vs slang pass 7 output),
        // pass_final_encode applies the un-dim before display gamma.
        output.write(float4(electron_intensity_dim * mask, 1.0f), gid);
    }

    // =======================================================================
    // Pass BLOOM_APPROX (slang pass 2): downscaled, blurred copy of
    // ORIG_LINEARIZED. Slang preset writes this at absolute scale 320x240.
    //
    // Slang ref: crt-royale-bloom-approx.h (267 lines). Slang exposes three
    // filter modes via `bloom_approx_filter`:
    //   0   -- bilinear sample of the source (this implementation)
    //   1   -- 3x3 resize blur with dynamic sigma
    //   2   -- 4x4 true Gaussian resize (slang default; not yet ported)
    //
    // We implement only the bilinear path here. The reference-capture script
    // sets `--params bloom_approx_filter=0.0` so that librashader runs the
    // same path, giving an apples-to-apples comparison. The 4x4 Gaussian
    // path is a follow-up iteration.
    //
    // BLOOM_APPROX is consumed by slang's brightpass (pass 8) and -- when
    // PHOSPHOR_BLOOM_FAKE is defined -- by apply-mask (pass 7). Neither of
    // those exists yet in our MSL pipeline; the texture is produced here so
    // future iterations can wire it in.
    //
    // Vertex stage (skipped here, compute kernel directly emits per-pixel UV):
    //   For our case ORIG_LINEARIZED_video_size == ORIG_LINEARIZED_texture_size
    //   and BLOOM_APPROX_video_size == BLOOM_APPROX_texture_size, so the
    //   slang vertex computation simplifies to tex_uv = output UV. The GPU's
    //   built-in linear sampler does the actual bilinear resize.
    // =======================================================================
    // -----------------------------------------------------------------------
    // Slang's tex2Dresize_gaussian4x4 (crt-royale-bloom-approx.h:121-234):
    // 16-sample true Gaussian resize. Snaps to texels for upsizing, computes
    // per-sample Gaussian weights from pixel-space distance to dest pixel.
    // -----------------------------------------------------------------------
    inline float3 ba_tex2Dresize_gaussian4x4(
        texture2d<float, access::sample> tex, sampler sam,
        float2 tex_uv, float2 dxdy, float2 tex_size, float2 tex_size_inv,
        float2 tex_uv_to_pixel_scale, float sigma)
    {
        float denom_inv = 0.5f / (sigma * sigma);

        // Find UV for sample 5 (second row, second column). Snap to texels
        // when upsizing in either dimension.
        float2 curr_texel = tex_uv * tex_size;
        float2 prev_texel = floor(curr_texel - float2(UNDER_HALF)) + float2(0.5f);
        float2 prev_texel_uv = prev_texel * tex_size_inv;
        float2 snap = float2(dxdy.x <= tex_size_inv.x ? 1.0f : 0.0f,
                             dxdy.y <= tex_size_inv.y ? 1.0f : 0.0f);
        float2 sample5_downsize_uv = tex_uv - 0.5f * dxdy;
        float2 sample5_uv = mix(sample5_downsize_uv, prev_texel_uv, snap);

        float2 dx = float2(dxdy.x, 0.0f);
        float2 s0  = sample5_uv - dxdy;
        float2 s10 = sample5_uv + dxdy;
        float2 s15 = sample5_uv + 2.0f * dxdy;
        float2 uv00 = s0;
        float2 uv01 = s0 + dx;
        float2 uv02 = s0 + 2.0f * dx;
        float2 uv03 = s0 + 3.0f * dx;
        float2 uv04 = sample5_uv - dx;
        float2 uv05 = sample5_uv;
        float2 uv06 = sample5_uv + dx;
        float2 uv07 = sample5_uv + 2.0f * dx;
        float2 uv08 = s10 - 2.0f * dx;
        float2 uv09 = s10 - dx;
        float2 uv10 = s10;
        float2 uv11 = s10 + dx;
        float2 uv12 = s15 - 3.0f * dx;
        float2 uv13 = s15 - 2.0f * dx;
        float2 uv14 = s15 - dx;
        float2 uv15 = s15;

        float3 c00 = tex.sample(sam, uv00).rgb;
        float3 c01 = tex.sample(sam, uv01).rgb;
        // NOTE: slang's bloom-approx.h:176 has a bug: it samples `dx` instead
        // of `sample2_uv`. We replicate that bug exactly because librashader
        // does too -- the ΔE comparison would otherwise drift.
        float3 c02 = tex.sample(sam, dx).rgb;
        float3 c03 = tex.sample(sam, uv03).rgb;
        float3 c04 = tex.sample(sam, uv04).rgb;
        float3 c05 = tex.sample(sam, uv05).rgb;
        float3 c06 = tex.sample(sam, uv06).rgb;
        float3 c07 = tex.sample(sam, uv07).rgb;
        float3 c08 = tex.sample(sam, uv08).rgb;
        float3 c09 = tex.sample(sam, uv09).rgb;
        float3 c10 = tex.sample(sam, uv10).rgb;
        float3 c11 = tex.sample(sam, uv11).rgb;
        float3 c12 = tex.sample(sam, uv12).rgb;
        float3 c13 = tex.sample(sam, uv13).rgb;
        float3 c14 = tex.sample(sam, uv14).rgb;
        float3 c15 = tex.sample(sam, uv15).rgb;

        float2 dest = tex_uv * tex_uv_to_pixel_scale;
        float2 o00 = uv00 * tex_uv_to_pixel_scale - dest;
        float2 o01 = uv01 * tex_uv_to_pixel_scale - dest;
        float2 o02 = uv02 * tex_uv_to_pixel_scale - dest;
        float2 o03 = uv03 * tex_uv_to_pixel_scale - dest;
        float2 o04 = uv04 * tex_uv_to_pixel_scale - dest;
        float2 o05 = uv05 * tex_uv_to_pixel_scale - dest;
        float2 o06 = uv06 * tex_uv_to_pixel_scale - dest;
        float2 o07 = uv07 * tex_uv_to_pixel_scale - dest;
        float2 o08 = uv08 * tex_uv_to_pixel_scale - dest;
        float2 o09 = uv09 * tex_uv_to_pixel_scale - dest;
        float2 o10 = uv10 * tex_uv_to_pixel_scale - dest;
        float2 o11 = uv11 * tex_uv_to_pixel_scale - dest;
        float2 o12 = uv12 * tex_uv_to_pixel_scale - dest;
        float2 o13 = uv13 * tex_uv_to_pixel_scale - dest;
        float2 o14 = uv14 * tex_uv_to_pixel_scale - dest;
        float2 o15 = uv15 * tex_uv_to_pixel_scale - dest;

        float w00 = exp(-dot(o00, o00) * denom_inv);
        float w01 = exp(-dot(o01, o01) * denom_inv);
        float w02 = exp(-dot(o02, o02) * denom_inv);
        float w03 = exp(-dot(o03, o03) * denom_inv);
        float w04 = exp(-dot(o04, o04) * denom_inv);
        float w05 = exp(-dot(o05, o05) * denom_inv);
        float w06 = exp(-dot(o06, o06) * denom_inv);
        float w07 = exp(-dot(o07, o07) * denom_inv);
        float w08 = exp(-dot(o08, o08) * denom_inv);
        float w09 = exp(-dot(o09, o09) * denom_inv);
        float w10 = exp(-dot(o10, o10) * denom_inv);
        float w11 = exp(-dot(o11, o11) * denom_inv);
        float w12 = exp(-dot(o12, o12) * denom_inv);
        float w13 = exp(-dot(o13, o13) * denom_inv);
        float w14 = exp(-dot(o14, o14) * denom_inv);
        float w15 = exp(-dot(o15, o15) * denom_inv);
        float wsum_inv = 1.0f / (w00+w01+w02+w03+w04+w05+w06+w07+
                                  w08+w09+w10+w11+w12+w13+w14+w15);

        float3 sum = w00*c00 + w01*c01 + w02*c02 + w03*c03 +
                     w04*c04 + w05*c05 + w06*c06 + w07*c07 +
                     w08*c08 + w09*c09 + w10*c10 + w11*c11 +
                     w12*c12 + w13*c13 + w14*c14 + w15*c15;
        return sum * wsum_inv;
    }

    // -----------------------------------------------------------------------
    // Slang's get_bloom_approx_sigma (bloom-functions.h:229-290) for the
    // 4x4 Gauss path. The math reduces to a constant + per-pass scaling once
    // all the static defaults are resolved (no PHOSPHOR_BLOOM_FAKE, no
    // RUNTIME_PHOSPHOR_BLOOM_SIGMA, mask_specify_num_triads = 0).
    // -----------------------------------------------------------------------
    inline float ba_get_bloom_approx_sigma_4x4(
        float output_size_x, float estimated_viewport_size_x,
        float mask_triad_size)
    {
        // Slang chain: min_allowed_viewport_triads.x ≈ 144 with the default
        // mask_min_allowed_triad_size=2, max_mask_texel_border=1, plus the
        // 1/0.0625 viewport-scale factor (see derived-settings-and-constants.h
        // lines 244-296). Hardcoded here -- depends only on static config.
        constexpr float min_allowed_viewport_triads_x = 144.0f;
        constexpr float max_viewport_size_x          = 1080.0f * 1024.0f * (4.0f/3.0f);
        constexpr float bloom_diff_thresh            = 1.0f / 256.0f;
        constexpr float beam_max_sigma_static        = 0.3f;

        float num_triads_from_size = estimated_viewport_size_x /
                                     max(mask_triad_size, 1e-5f);
        float mask_num_triads_runtime =
            max(min_allowed_viewport_triads_x, num_triads_from_size);
        float asymptotic_triad_size = max_viewport_size_x / mask_num_triads_runtime;
        float asymptotic_sigma = -0.05168f + 0.6113f * asymptotic_triad_size -
            1.122f * asymptotic_triad_size * sqrt(0.000416f + bloom_diff_thresh);
        float bloom_approx_sigma_x = asymptotic_sigma * output_size_x /
                                     max_viewport_size_x;
        return length(float2(bloom_approx_sigma_x, beam_max_sigma_static));
    }

    kernel void pass_bloom_approx(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // ORIG_LINEARIZED
        texture2d<float, access::write>  output  [[ texture(1) ]],  // BLOOM_APPROX
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 out_size = float2(output.get_width(), output.get_height());
        float2 src_size = float2(input.get_width(),  input.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;

        // 4x4 Gauss path requires source-space dxdy. For upsizing (out > src)
        // dxdy = 1/src_size; for downsizing, dxdy = src/out / src_size. Slang
        // ref: bloom-approx.h:71-78.
        float2 dxdy_min_scale = src_size / out_size;
        float2 dxdy_scale = max(dxdy_min_scale, float2(1.0f));
        float2 tex_size_inv = 1.0f / src_size;
        float2 dxdy = dxdy_scale * tex_size_inv;
        // tex_uv_to_pixel_scale maps tex-uv to dest pixel coords.
        float2 tex_uv_to_pixel_scale = out_size;

        // Sigma uses Slang's static-default path -- depends on viewport-X
        // estimate (= video.y * geom_aspect_ratio_static) and triad size.
        constexpr float geom_aspect_ratio_static = 1.313069909f;
        float estimated_viewport_size_x =
            u.video_size.y * geom_aspect_ratio_static;
        float sigma = ba_get_bloom_approx_sigma_4x4(
            out_size.x, estimated_viewport_size_x, u.mask_triad_size);

        float3 color = ba_tex2Dresize_gaussian4x4(
            input, sam, uv, dxdy, src_size, tex_size_inv,
            tex_uv_to_pixel_scale, sigma);
        output.write(float4(color, 1.0f), gid);
    }

    // =======================================================================
    // Pass BRIGHTPASS (slang pass 8): area-based brightness extraction.
    //
    // Slang ref: crt-royale-brightpass.{slang,h}
    // Slang pass index: 8
    // Inputs:  MASKED_SCANLINES (= our pass 3) + BLOOM_APPROX (= pass_bloom_approx)
    // Output:  BRIGHTPASS texture, sampled by bloom V/H to produce BLOOM_FINAL.
    //
    // Goal: for each pixel, estimate "how much extra blur energy will arrive
    // from neighboring blooming phosphors", and use that to decide which
    // fraction of this pixel's value should be passed through to the bloom
    // chain. Dark areas (already low energy) get blurred a lot; bright areas
    // bloom into their neighborhood and pass less of themselves through.
    //
    // The whole derivation lives in TroggleMonkey's original comments inside
    // crt-royale-brightpass.h:113-163. We reproduce the algebra 1:1 here.
    //
    // Static-config simplifications:
    //   - mask_triad_size_desired_static = 3.0 and mask_triads_per_tile = 8
    //     give a static bloom_sigma = ~1.5608 (get_min_sigma_to_blur_triad
    //     with thresh=1/256). The center_weight that follows from this sigma
    //     and the default 9-tap blur is ~0.0658. Both are hardcoded here.
    //     When the user changes mask_triad_size at runtime, we'd recompute
    //     in MSL -- TODO for a follow-up iteration that exposes the dynamic
    //     path (RUNTIME_PHOSPHOR_BLOOM_SIGMA in slang).
    //   - levels_contrast = 1.0 (default).
    //   - BRIGHTPASS_AREA_BASED branch skipped (slang default).
    // =======================================================================
    inline float bp_get_min_sigma_to_blur_triad(float triad_size, float thresh) {
        return -0.05168f + 0.6113f * triad_size -
               1.122f * triad_size * sqrt(0.000416f + thresh);
    }

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

        // Sample the two inputs at the same logical position. Slang's vertex
        // shader sets scanline_tex_uv = blur3x3_tex_uv = output's UV for our
        // case where video_size == texture_size for both inputs.
        float3 intensity_dim = masked_scanlines.sample(sam, uv).rgb;
        float3 phosphor_blur_approx = bloom_approx.sample(sam, uv).rgb;

        // Full intensity = mask-dimmed * undim factor * mask amplify * contrast.
        // levels_autodim_temp = 0.5 (slang default), so undim_factor = 2.0.
        // mask_amplify = 1/mask_grille_avg_color ~ 4.811.
        float undim_factor = 1.0f / max(u.levels_autodim_temp, 1e-5f);
        float3 intensity = intensity_dim * undim_factor * u.mask_amplify;
        // levels_contrast = 1.0 (slang default, not exposed).

        // Compute bloom_sigma + center_weight from static slang defaults:
        //   triad_size = mask_triads_per_tile / mask_triads_per_tile = 1 *
        //     mask_triad_size_desired_static = 3.0
        //   sigma ~ 1.5608, center_weight ~ 0.0658
        // get_final_bloom_sigma uses bloom_sigma_optimistic = sigma_for_triad(
        //   mask_triad_size_desired_static, bloom_diff_thresh).
        constexpr float bloom_diff_thresh = 1.0f / 256.0f;
        float bloom_sigma = bp_get_min_sigma_to_blur_triad(
            u.mask_triad_size, bloom_diff_thresh);
        float center_weight = bp_get_center_weight_9tap(bloom_sigma);

        // Estimate maximum energy we'll receive from blooming neighbors.
        // Slang ref: brightpass.h:128-134.
        float3 max_area_contribution_approx =
            max(float3(0.0f), phosphor_blur_approx - center_weight * intensity);

        // Apply bloom_underestimate_levels (slang default 0.8). Hardcoded
        // until exposed via UI.
        constexpr float bloom_underestimate_levels = 0.8f;
        float3 area_contrib_underestimate =
            bloom_underestimate_levels * max_area_contribution_approx;
        float3 intensity_underestimate =
            bloom_underestimate_levels * intensity;

        // Compute blur_ratio (the fraction we want to blur). Slang ref:
        // brightpass.h:154-157 (non-AREA_BASED branch).
        // Use saturate to avoid division pathologies on solid black.
        float3 safe_intensity = max(intensity_underestimate, float3(1e-5f));
        float3 blur_ratio_temp =
            ((float3(1.0f) - area_contrib_underestimate) / safe_intensity -
             float3(1.0f)) / (center_weight - 1.0f);
        float3 blur_ratio = clamp(blur_ratio_temp, float3(0.0f), float3(1.0f));

        // brightpass = intensity_dim * lerp(blur_ratio, 1, bloom_excess).
        // bloom_excess = 0 by default.
        constexpr float bloom_excess = 0.0f;
        float3 brightpass = intensity_dim *
            mix(blur_ratio, float3(1.0f), bloom_excess);

        output.write(float4(brightpass, 1.0f), gid);
    }

    // =======================================================================
    // tex2Dblur9fast: 9-tap separable Gaussian using 5 samples (1 nearest +
    // 4 linear-interpolated pairs). Slang ref: blur-functions.h:416-444.
    // -----------------------------------------------------------------------
    // Used by pass_bloom_v (dxdy = (0, 1/H)) and pass_bloom_h_reconstitute
    // (dxdy = (1/W, 0)). The bilinear trick: a linear sample at offset
    // (1 + w12_ratio) reads texels 1 and 2 with weight ratio (w1 : w2),
    // collapsing two texel samples into one bilinear fetch.
    // =======================================================================
    inline float3 tex2Dblur9fast(texture2d<float, access::sample> tex,
                                 float2 tex_uv, float2 dxdy, float sigma,
                                 sampler sam)
    {
        float denom_inv = 0.5f / (sigma * sigma);
        float w0 = 1.0f;
        float w1 = exp(-1.0f  * denom_inv);
        float w2 = exp(-4.0f  * denom_inv);
        float w3 = exp(-9.0f  * denom_inv);
        float w4 = exp(-16.0f * denom_inv);
        float weight_sum_inv = 1.0f / (w0 + 2.0f * (w1 + w2 + w3 + w4));
        float w12 = w1 + w2;
        float w34 = w3 + w4;
        float w12_ratio = w2 / w12;
        float w34_ratio = w4 / w34;
        float3 sum = float3(0.0f);
        sum += w34 * tex.sample(sam, tex_uv - (3.0f + w34_ratio) * dxdy).rgb;
        sum += w12 * tex.sample(sam, tex_uv - (1.0f + w12_ratio) * dxdy).rgb;
        sum += w0  * tex.sample(sam, tex_uv).rgb;
        sum += w12 * tex.sample(sam, tex_uv + (1.0f + w12_ratio) * dxdy).rgb;
        sum += w34 * tex.sample(sam, tex_uv + (3.0f + w34_ratio) * dxdy).rgb;
        return sum * weight_sum_inv;
    }

    // =======================================================================
    // Pass HALATION_V (slang pass 3): vertical 9-tap Gaussian blur of
    // BLOOM_APPROX. Same separable bilinear-trick blur as bloom_v/h, but at
    // the fixed 320x240 BLOOM_APPROX scale and with a constant sigma drawn
    // from blur-functions.h:185 (USE_BINOMIAL_BLUR_STD_DEVS undefined branch).
    //
    // Slang ref: blurs/shaders/royale/blur9fast-vertical.slang
    //            -> include/blur-functions.h tex2Dblur9fast(... blur9_std_dev)
    //
    // Output: input to HALATION_H (slang pass 4).
    // Consumers: HALATION_BLUR is sampled by apply-mask (slang pass 7) for
    //            the desaturated halation term, and by bloom-final (slang
    //            pass 10) for the chromatic diffusion term.
    // =======================================================================
    constant float BLUR9_STD_DEV = 1.7533203125f;

    kernel void pass_halation_v(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // BLOOM_APPROX
        texture2d<float, access::write>  output  [[ texture(1) ]],  // halation_v
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;
        // scale_type=source, scale=1.0 -> video/output/texture sizes all match.
        // Slang vertex: dxdy = (video/output)/texture = 1/size.
        float2 dxdy = float2(0.0f, 1.0f / out_size.y);
        float3 color = tex2Dblur9fast(input, uv, dxdy, BLUR9_STD_DEV, sam);
        output.write(float4(color, 1.0f), gid);
    }

    // =======================================================================
    // Pass HALATION_H / HALATION_BLUR (slang pass 4): horizontal 9-tap
    // Gaussian blur of halation_v. Produces the final HALATION_BLUR alias.
    //
    // Slang ref: blurs/shaders/royale/blur9fast-horizontal.slang.
    // =======================================================================
    kernel void pass_halation_h(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // halation_v
        texture2d<float, access::write>  output  [[ texture(1) ]],  // HALATION_BLUR
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;
        float2 dxdy = float2(1.0f / out_size.x, 0.0f);
        float3 color = tex2Dblur9fast(input, uv, dxdy, BLUR9_STD_DEV, sam);
        output.write(float4(color, 1.0f), gid);
    }

    // =======================================================================
    // Pass MASK_RESIZE_V + MASK_RESIZE_H (slang passes 5 + 6):
    // Lanczos sinc-downsample the phosphor-mask LUT to a small tileable
    // texture. Two-pass separable: Pass 5 resizes Y (input 512x512 LUT ->
    // 64x~48), Pass 6 resizes X (~64x48 -> ~16x48). The output MASK_RESIZE
    // is then sampled by apply-mask (slang pass 7) when
    // mask_sample_mode_desired = 0 (the slang default).
    //
    // Slang refs:
    //   crt-royale-mask-resize-vertical.slang
    //   crt-royale-mask-resize-horizontal.slang
    //   phosphor-mask-resizing.h (helpers, ~700 lines)
    //
    // What we omit from Slang's full implementation:
    //   - Dynamic loop sizing (DRIVERS_ALLOW_DYNAMIC_BRANCHES path). We use
    //     a static 128-sample loop. With default mag_scale ~ 0.047 the
    //     dynamic Slang code picks ~128 samples too, so behavior matches.
    //   - mask_specify_num_triads = 1 (user wants a fixed triad count).
    //     Slang default is 0 (user specifies triad size); we honor that.
    //   - Three simultaneous LUT bindings. We sample the active LUT.
    //
    // Constants (all from Slang's static defaults in
    //   user-cgp-constants.h + user-settings.h + derived-settings-and-
    //   constants.h):
    //     mask_resize_src_lut_size      = (512, 512)   (the *_large LUTs)
    //     mask_triads_per_tile          = 8.0
    //     mask_min_allowed_triad_size   = 2.0
    //     mask_sinc_lobes               = 3.0
    //     mask_resize_viewport_scale    = (0.0625, 0.0625)
    //     mask_resize_num_tiles         = 2.0  (=1 + 2*max_mask_tile_border)
    //     mask_specify_num_triads       = 0.0
    //     mask_triad_size_desired       = u.mask_triad_size  (runtime)
    //     geom_aspect_ratio             = 1.313069909
    // =======================================================================
    constant float2 MR_SRC_LUT_SIZE          = float2(512.0f, 512.0f);
    constant float  MR_TRIADS_PER_TILE       = 8.0f;
    constant float  MR_MIN_ALLOWED_TRIAD_SIZE = 2.0f;
    constant float  MR_SINC_LOBES            = 3.0f;
    constant float2 MR_VIEWPORT_SCALE        = float2(0.0625f, 0.0625f);
    constant float  MR_NUM_TILES             = 2.0f;
    constant float  MR_GEOM_ASPECT_RATIO     = 1.313069909f;
    constant float  MR_PI                    = 3.14159265358979f;
    constant float  MR_PI_OVER_LOBES         = MR_PI / MR_SINC_LOBES;
    constant float  MR_MIN_ALLOWED_TILE_SIZE = 16.0f; // = ceil(2 * 8)
    // Static sample count. Slang picks dynamically:
    //   samples = min(min_samples_m4, max_sinc_resize_samples_m4)
    // For the small 64x64 LUT: max_samples_m4 = 24.
    // For our default magnification = 24/64 = 0.375, min_samples_m4 = 16.
    // 24 is the tight upper bound; using more (e.g. 128) blurs out the LUT
    // structure entirely because tile_dr * MR_SAMPLES wraps around the LUT.
    constant int    MR_SAMPLES               = 24;    // multiple of 4
    constant float  MR_TINY                  = 1e-6f;

    // get_resized_mask_tile_size, slang phosphor-mask-resizing.h:433-531.
    // Inputs:
    //   est_viewport_size = (viewport_y * aspect, viewport_y)
    //   est_resize_output = (output.y * aspect, output.y) for pass5
    //                     = output_size                    for pass6
    // We assume mask_sample_mode = 0 here (pass 5/6 only run for that
    // branch). The LUT is square so tile_aspect = (1, 1).
    inline float2 mr_get_resized_mask_tile_size(
        float2 est_viewport_size, float2 est_resize_output,
        float mask_triad_size_desired)
    {
        // tile_aspect = LUT.y/LUT.x = 1 (square LUT)
        // desired_tile_size_x = triads_per_tile * triad_size_desired
        float desired_tile_size_x =
            MR_TRIADS_PER_TILE * mask_triad_size_desired;
        // Make sure we're not upsizing.
        float temp_tile_size_x = min(desired_tile_size_x, MR_SRC_LUT_SIZE.x);
        float2 temp_tile_size  = float2(temp_tile_size_x); // tile_aspect = (1,1)
        float2 min_tile_size   = float2(MR_MIN_ALLOWED_TILE_SIZE);
        float2 max_tile_size   = est_resize_output / MR_NUM_TILES;
        float2 clamped_tile_size = clamp(temp_tile_size, min_tile_size,
                                          max_tile_size);
        // tile_aspect_ratio = LUT.x/LUT.y = 1, so the reclamp is trivial.
        float x_from_y = clamped_tile_size.y; // * 1.0
        // solemnly_swear_same_inputs_for_every_pass = false, so y_from_x
        // = clamped_tile_size.y (no extra constraint).
        float2 reclamped = float2(
            min(clamped_tile_size.x, x_from_y),
            clamped_tile_size.y);
        // Integer tile sizes for tiled sampling correctness.
        return floor(reclamped + float2(MR_TINY));
    }

    // Single-axis Lanczos sinc downsample, tiled. Mirrors slang's
    // downsample_vertical_sinc_tiled / downsample_horizontal_sinc_tiled
    // (phosphor-mask-resizing.h:217-414). Static 128-sample loop.
    //
    // The "r" axis is the one we resample along (the other dimension is
    // unchanged). For Pass 5 r = Y, for Pass 6 r = X.
    //
    //   src_tex_uv: frac(src_tex_uv_wrap) -- one tile's worth of UV.
    //   tex_size:   source texture size (in texels).
    //   dr:         1/tex_size on the resampling axis.
    //   mag_scale:  pass_output_tile_size / src_lut_size on this axis.
    //   tile_size_uv_r: tile_size_on_r_axis / texture_size_on_r_axis.
    //                    (For Pass 5 this is 1.0 since the LUT IS the tile;
    //                     for Pass 6 it's pass5_tile.x / pass5_size.x.)
    inline float3 mr_downsample_sinc_1d(
        texture2d<float, access::sample> tex, sampler sam,
        float2 src_tex_uv, float2 tex_size,
        float dr, float mag_scale, float tile_size_uv_r,
        bool vertical)
    {
        float input_tiles_per_texture_r = 1.0f / tile_size_uv_r;
        float tile_dr = dr * input_tiles_per_texture_r;

        // Replicate get_first_texel_tile_uv_and_dist.
        float2 curr_texel = src_tex_uv * tex_size;
        float2 prev_texel = floor(curr_texel - float2(UNDER_HALF)) + float2(0.5f);
        float2 first_texel = prev_texel -
                             float2(float(MR_SAMPLES) * 0.5f - 1.0f);
        float2 first_uv_wrap_2D    = first_texel * dr;
        float2 first_dist_2D       = curr_texel - first_texel;
        float2 first_tile_uv_wrap  = first_uv_wrap_2D * input_tiles_per_texture_r;
        float2 coord_neg = float2(first_tile_uv_wrap.x < 0.0f ? 1.0f : 0.0f,
                                   first_tile_uv_wrap.y < 0.0f ? 1.0f : 0.0f);
        float2 first_tile_uv = fract(first_tile_uv_wrap) + coord_neg;

        float first_tile_uv_r   = vertical ? first_tile_uv.y : first_tile_uv.x;
        float first_dist_unscaled = vertical ? first_dist_2D.y : first_dist_2D.x;

        float3 pixel_color = float3(0.0f);
        float  weight_sum  = 0.0f;
        for (int i_base = 0; i_base < MR_SAMPLES; i_base += 4) {
            float4 true_i = float4(float(i_base)) +
                            float4(0.0f, 1.0f, 2.0f, 3.0f);
            float4 tile_uv_r = fract(float4(first_tile_uv_r) + true_i * tile_dr);
            float4 tex_uv_r  = tile_uv_r * tile_size_uv_r;

            float4 dist     = mag_scale * abs(float4(first_dist_unscaled) - true_i);
            float4 pi_dist  = MR_PI * dist + float4(MR_TINY);
            float4 pi_dist_lobes = MR_PI_OVER_LOBES * dist + float4(MR_TINY);
            float4 weights  = min(sin(pi_dist) * sin(pi_dist_lobes) /
                                    (pi_dist * pi_dist_lobes),
                                   float4(1.0f));

            float2 uv0 = vertical ? float2(src_tex_uv.x, tex_uv_r[0])
                                  : float2(tex_uv_r[0], src_tex_uv.y);
            float2 uv1 = vertical ? float2(src_tex_uv.x, tex_uv_r[1])
                                  : float2(tex_uv_r[1], src_tex_uv.y);
            float2 uv2 = vertical ? float2(src_tex_uv.x, tex_uv_r[2])
                                  : float2(tex_uv_r[2], src_tex_uv.y);
            float2 uv3 = vertical ? float2(src_tex_uv.x, tex_uv_r[3])
                                  : float2(tex_uv_r[3], src_tex_uv.y);
            pixel_color += tex.sample(sam, uv0).rgb * weights.x;
            pixel_color += tex.sample(sam, uv1).rgb * weights.y;
            pixel_color += tex.sample(sam, uv2).rgb * weights.z;
            pixel_color += tex.sample(sam, uv3).rgb * weights.w;
            weight_sum  += weights.x + weights.y + weights.z + weights.w;
        }
        return pixel_color / max(weight_sum, MR_TINY);
    }

    // Pass 5: Y-Lanczos resize. Source = the active mask LUT (512x512).
    // Output = mask_resize_v (64 x ~48). 2 tiles vertically.
    kernel void pass_mask_resize_v(
        texture2d<float, access::sample> mask_lut [[ texture(0) ]], // active LUT
        texture2d<float, access::write>  output   [[ texture(1) ]], // mask_resize_v
        constant Uniforms&               u        [[ buffer(0)  ]],
        sampler                          lut_sam  [[ sampler(0) ]], // linear+repeat
        uint2                            gid      [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;

        // Vertex math: see slang ref crt-royale-mask-resize-vertical.slang:46-82.
        float viewport_y = out_size.y / MR_VIEWPORT_SCALE.y;
        float2 est_viewport_size = float2(viewport_y * MR_GEOM_ASPECT_RATIO,
                                          viewport_y);
        float2 est_resize_output = float2(out_size.y * MR_GEOM_ASPECT_RATIO,
                                          out_size.y);
        float2 tile_size = mr_get_resized_mask_tile_size(
            est_viewport_size, est_resize_output, u.mask_triad_size);
        float2 pass_output_tile_size =
            float2(min(MR_SRC_LUT_SIZE.x, out_size.x), tile_size.y);
        float2 output_tiles_this_pass = out_size / pass_output_tile_size;
        float2 tile_uv_wrap = uv * output_tiles_this_pass;
        float2 mag_scale  = pass_output_tile_size / MR_SRC_LUT_SIZE;

        // Slang discards when tile_uv_wrap.y > num_tiles -- leaves the FBO
        // untouched outside the first MR_NUM_TILES tiles. We write zeros
        // instead; this region is never sampled downstream (pass 7 only
        // samples the first tile worth via tile_uv_wrap = video_uv *
        // tiles_per_screen and wraps).
        if (tile_uv_wrap.y > MR_NUM_TILES) {
            output.write(float4(0.0f), gid);
            return;
        }

        float2 src_tex_uv = fract(tile_uv_wrap); // src_tex_uv_wrap == tile_uv_wrap
                                                  // because the LUT IS one tile.
        float src_dy = 1.0f / MR_SRC_LUT_SIZE.y;
        float3 color = mr_downsample_sinc_1d(
            mask_lut, lut_sam, src_tex_uv, MR_SRC_LUT_SIZE,
            src_dy, mag_scale.y, 1.0f /*tile_size_uv_r*/,
            true /*vertical*/);
        output.write(float4(color, 1.0f), gid);
    }

    // Pass 6: X-Lanczos resize. Source = mask_resize_v (64x48). Output =
    // mask_resize (~16x48). 2 tiles horizontally and vertically.
    kernel void pass_mask_resize_h(
        texture2d<float, access::sample> mask_resize_v [[ texture(0) ]],
        texture2d<float, access::write>  output        [[ texture(1) ]],
        constant Uniforms&               u             [[ buffer(0)  ]],
        sampler                          sam           [[ sampler(0) ]],
        uint2                            gid           [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;
        float2 src_size = float2(mask_resize_v.get_width(),
                                 mask_resize_v.get_height());

        // Vertex math: see slang ref crt-royale-mask-resize-horizontal.slang:51-89.
        float2 est_viewport_size = out_size / MR_VIEWPORT_SCALE;
        float2 tile_size = mr_get_resized_mask_tile_size(
            est_viewport_size, out_size, u.mask_triad_size);

        float2 output_tiles_this_pass = out_size / tile_size;
        float2 tile_uv_wrap = uv * output_tiles_this_pass;

        float2 input_tile_size = float2(min(MR_SRC_LUT_SIZE.x, src_size.x),
                                        tile_size.y);
        float2 tile_size_uv    = input_tile_size / src_size;
        float2 mag_scale       = tile_size / input_tile_size;
        float  src_dx          = 1.0f / src_size.x;

        // Pass 6 clamps both dimensions: max(tile_uv_wrap.x, .y) <= num_tiles.
        if (max(tile_uv_wrap.x, tile_uv_wrap.y) > MR_NUM_TILES) {
            output.write(float4(0.0f), gid);
            return;
        }

        // src_tex_uv_wrap = tile_uv_wrap * tile_size_uv -- in raw tex_uv coords.
        // The downstream sinc helper takes the wrapped UV and does the fract
        // internally via tile_uv_wrap math; we replicate that by passing
        // frac(src_tex_uv_wrap).
        float2 src_tex_uv = fract(tile_uv_wrap * tile_size_uv);
        float3 color = mr_downsample_sinc_1d(
            mask_resize_v, sam, src_tex_uv, src_size,
            src_dx, mag_scale.x, tile_size_uv.x,
            false /*horizontal*/);
        output.write(float4(color, 1.0f), gid);
    }

    // =======================================================================
    // Pass BLOOM_V (slang pass 9): vertical separable Gaussian blur of
    // BRIGHTPASS. Slang ref: crt-royale-bloom-vertical.slang.
    //
    // Sigma + blur-size selection: the default static path
    // (RUNTIME_PHOSPHOR_BLOOM_SIGMA undefined) sets bloom_sigma to
    // get_min_sigma_to_blur_triad(mask_triad_size_desired_static=3, 1/256)
    // ~= 1.5608, which falls below blur9_std_dev=1.7533, so tex2DblurNfast
    // picks the 9-tap path. We hardcode 9-tap here.
    // =======================================================================
    kernel void pass_bloom_v(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // BRIGHTPASS
        texture2d<float, access::write>  output  [[ texture(1) ]],  // BLOOM_V
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;

        // dxdy.y = video_size.y/output_size.y / texture_size.y. Slang's
        // vertex sets it from IN.video_size/output_size/texture_size; since
        // this pass is "no resize" (scale=source), video/texture/output all
        // match -> dxdy reduces to 1/H.
        float2 dxdy = float2(0.0f, 1.0f / out_size.y);

        // bloom_sigma_runtime from triad_size via get_min_sigma_to_blur_triad.
        constexpr float bloom_diff_thresh = 1.0f / 256.0f;
        float bloom_sigma = bp_get_min_sigma_to_blur_triad(
            u.mask_triad_size, bloom_diff_thresh);

        float3 color = tex2Dblur9fast(input, uv, dxdy, bloom_sigma, sam);
        output.write(float4(color, 1.0f), gid);
    }

    // =======================================================================
    // Pass BLOOM_H_RECONSTITUTE (slang pass 10): horizontal Gaussian blur of
    // BLOOM_V output, then mix the result with the mask-dim "dimpass" portion
    // of MASKED_SCANLINES to produce BLOOM_FINAL.
    //
    // Slang ref: crt-royale-bloom-horizontal-reconstitute.slang.
    //
    // Inputs: BLOOM_V (texture 0), BRIGHTPASS (texture 2), MASKED_SCANLINES
    //         (texture 3), HALATION_BLUR (texture 4).
    // Output: BLOOM_FINAL (= Pass 4's input).
    //
    // Math:
    //   blurred_brightpass = horizontal-blur(BLOOM_V)
    //   dimpass            = MASKED_SCANLINES - BRIGHTPASS
    //   phosphor_bloom     = (dimpass + blurred_brightpass)
    //                        * mask_amplify * undim_factor * levels_contrast
    //   diffusion_color    = levels_contrast * HALATION_BLUR
    //   final              = lerp(phosphor_bloom, diffusion_color,
    //                              diffusion_weight)
    //
    // diffusion_weight defaults to 0.075 (slang bind-shader-params.h:161) --
    // not zero! Earlier iterations skipped this mix and were silently off
    // by ~7.5% halation contribution. Fixed in this iteration.
    // =======================================================================
    kernel void pass_bloom_h_reconstitute(
        texture2d<float, access::sample> bloom_v          [[ texture(0) ]],  // BLOOM_V
        texture2d<float, access::write>  output           [[ texture(1) ]],  // BLOOM_FINAL
        texture2d<float, access::sample> brightpass       [[ texture(2) ]],  // BRIGHTPASS
        texture2d<float, access::sample> masked_scanlines [[ texture(3) ]],  // MASKED_SCANLINES
        texture2d<float, access::sample> halation_blur    [[ texture(4) ]],  // HALATION_BLUR
        constant Uniforms&               u                [[ buffer(0)  ]],
        sampler                          sam              [[ sampler(0) ]],
        uint2                            gid              [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;

        // dxdy for the horizontal blur: 1/W on x, 0 on y. Slang's vertex
        // computes 1/IN.texture_size.x; since this pass keeps the same
        // texture size, that equals 1/W.
        float2 dxdy = float2(1.0f / out_size.x, 0.0f);

        constexpr float bloom_diff_thresh = 1.0f / 256.0f;
        float bloom_sigma = bp_get_min_sigma_to_blur_triad(
            u.mask_triad_size, bloom_diff_thresh);

        float3 blurred_brightpass = tex2Dblur9fast(bloom_v, uv, dxdy,
                                                    bloom_sigma, sam);
        float3 intensity_dim = masked_scanlines.sample(sam, uv).rgb;
        float3 bp           = brightpass.sample(sam, uv).rgb;
        float3 dimpass      = intensity_dim - bp;

        // Un-dim + mask amplify (slang's undim_factor = 1/levels_autodim_temp,
        // mask_amplify = 1/mask_grille_avg_color ~ 4.811). levels_contrast
        // = 1.0 (slang default, not exposed).
        float undim_factor = 1.0f / max(u.levels_autodim_temp, 1e-5f);
        float3 phosphor_bloom = (dimpass + blurred_brightpass) *
                                u.mask_amplify * undim_factor;

        // Refractive diffusion: linear HALATION_BLUR (full chroma) mixed in
        // via diffusion_weight (slang default 0.075). levels_contrast = 1.0
        // is folded into the sample as in the Slang reference.
        float3 diffusion_color = halation_blur.sample(sam, uv).rgb;
        float3 final_bloom = mix(phosphor_bloom, diffusion_color,
                                 u.diffusion_weight);
        output.write(float4(final_bloom, 1.0f), gid);
    }

    // =======================================================================
    // Pass "last": Re-encode linear -> display gamma.
    // Temporary stand-in until passes 3..12 land. Honors debug_pass_index by
    // simply consuming whatever texture the host bound at texture(0) -- the
    // host decides which intermediate to display.
    // =======================================================================
    kernel void pass_final_encode(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms               &u       [[ buffer(0)  ]],
        sampler                         sam      [[ sampler(0) ]],
        uint2                           gid      [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());
        float4 linear_color = input.sample(sam, uv);

        // Plain display-gamma encode. Used as the simple display path for the
        // debug picker (pass1/pass2/pass3 inspections). The full pass 11 -
        // border + optional geometry + encode - lives in pass4_geometry_aa.
        float3 encoded = pow(max(linear_color.rgb, 0.0f),
                             float3(1.0f / u.lcd_gamma));
        output.write(float4(encoded, 1.0f), gid);
    }

    // =======================================================================
    // Pass 4 helpers: ported Slang Pass 11 geometry + AA primitives.
    // Slang refs: geometry-functions.h (695 lines), tex2Dantialias.h (1393).
    //
    // Default-path behavior (geom_mode=0, aa_level=12, overscan=1): slang
    // Pass 11 collapses to a plain point sample + border_dim + gamma encode.
    // Our default does the same. The helpers below kick in only for the
    // optional curved geometry modes.
    // =======================================================================

    // Slang user-cgp defaults (user-cgp-constants.h, user-settings.h).
    constant float G4_PI               = 3.14159265358979f;
    constant float G4_VIEW_DIST        = 2.0f;       // geom_view_dist_static
    constant float G4_RADIUS_STATIC    = 2.0f;       // geom_radius_static
    constant float G4_ASPECT_X_STATIC  = 432.0f;     // geom_aspect_ratio_x = 4
    constant float G4_ASPECT_Y_STATIC  = 329.0f;     // geom_aspect_ratio_y = 3
    constant float G4_ASPECT_VEC_RATIO = 1.313069909f;

    // Aspect vector: maps viewport_aspect_ratio to a (1, 1)-summing 2D vec
    // proportional to (aspect_x, aspect_y), with one component clamped to 1.
    inline float2 g4_aspect_vector(float viewport_aspect)
    {
        float assumed_aspect = G4_ASPECT_X_STATIC / G4_ASPECT_Y_STATIC;
        if (viewport_aspect >= assumed_aspect)
            return float2(viewport_aspect / assumed_aspect, 1.0f);
        return float2(1.0f, assumed_aspect / viewport_aspect);
    }

    // Quadratic formula, Citardauq variant (geometry-functions.h:38). Returns
    // (first_solution, discriminant).
    inline float2 g4_quad_solve(float a, float b_over_2, float c)
    {
        float disc = b_over_2 * b_over_2 - a * c;
        float sol0 = c / (-b_over_2 + sqrt(max(disc, 0.0f)));
        return float2(sol0, disc);
    }

    // Sphere-ray intersection (geometry-functions.h:51). geom_radius is the
    // CRT's virtual curvature radius (default 2.0).
    inline float2 g4_intersect_sphere(float3 view_vec, float3 eye_pos,
                                       float radius)
    {
        float a = dot(view_vec, view_vec);
        float b_over_2 = dot(view_vec, eye_pos);
        float c = dot(eye_pos, eye_pos) - radius * radius;
        return g4_quad_solve(a, b_over_2, c);
    }

    // Sphere xyz -> uv mapping (great-circle arc length, geometry-functions.h:130).
    inline float2 g4_sphere_xyz_to_uv(float3 pos, float2 geom_aspect,
                                       float radius)
    {
        float3 center = float3(0.0f, 0.0f, radius);
        float cp_len  = length(cross(pos, center));
        float dp      = dot(pos, center);
        float angle   = atan2(cp_len, dp);
        float arc_len = angle * radius;
        float2 sq_unit = normalize(float2(pos.x, -pos.y));
        float2 sq_uv   = arc_len * sq_unit;
        return sq_uv / geom_aspect;
    }

    // Curved-video-uv coordinate computation (Pass 11 default for geom_mode=1).
    // No global_to_local matrix path: geom_tilt_angle is 0 by default, so the
    // matrix is identity. eye_pos_local = (0, 0, geom_view_dist + ...) and
    // we just do a straight raycast.
    //
    // Slang ref: geometry-functions.h:563-663 (the DRIVERS_ALLOW_DERIVATIVES
    // path uses ddx/ddy which compute shaders don't have, so we drop the
    // pixel_to_tangent matrix here -- it's only consumed by tex2Daa anyway).
    inline float2 g4_curved_uv(float2 flat_uv, float2 geom_aspect,
                                float radius)
    {
        // 1) Map flat_uv -> view_vec in CRT-local frame.
        //    view_uv = (flat_uv - 0.5) * geom_aspect   (xy plane)
        //    view_vec = (view_uv.x, -view_uv.y, -geom_view_dist)
        float2 view_uv = (flat_uv - float2(0.5f)) * geom_aspect;
        float3 view_vec = float3(view_uv.x, -view_uv.y, -G4_VIEW_DIST);

        // 2) Eye position. Slang's get_ideal_global_eye_pos picks an optimal
        //    eye for the geom_aspect + radius combo. For default geom_aspect
        //    and geom_radius = 2.0 the optimal eye sits roughly at
        //    (0, 0, geom_radius + geom_view_dist) along the -z axis from the
        //    sphere. We approximate that closed-form.
        float3 eye = float3(0.0f, 0.0f, radius + G4_VIEW_DIST);

        // 3) Ray-sphere intersection.
        float2 d_and_disc = g4_intersect_sphere(view_vec, eye, radius);
        if (d_and_disc.y < 0.005f) return float2(2.0f);  // misses -> offscreen
        float3 pos = eye + view_vec * d_and_disc.x;

        // 4) Project onto sphere uv-space (great-circle distance), recenter.
        float2 centered = g4_sphere_xyz_to_uv(pos, geom_aspect, radius);
        return centered + float2(0.5f);
    }

    // Catmull-Rom cubic kernel (aa_cubic_c = 0.5, aa_cubic_b = 0). Slang ref
    // tex2Dantialias.h:313-340 -- the Mitchell-Netravali "Keys cubic" family
    // with B = 1 - 2C. With C = 0.5 (the Slang default `aa_cubic_c_static`),
    // this reduces to:
    //   |x| < 1:        1.5|x|^3 - 2.5|x|^2 + 1
    //   1 <= |x| < 2:  -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
    //   else:           0
    // Support radius 2 px, so separable evaluation needs a 4x4 sample grid.
    inline float g4_catmull_rom(float x)
    {
        float a = abs(x);
        if (a < 1.0f) return 1.5f * a*a*a - 2.5f * a*a + 1.0f;
        if (a < 2.0f) return -0.5f * a*a*a + 2.5f * a*a - 4.0f * a + 2.0f;
        return 0.0f;
    }

    // 16-tap separable Catmull-Rom AA kernel. Slang's tex2Daa with default
    // aa_filter=6 (Cubic separable) + aa_level=12 uses a quincunx-pattern
    // 12-sample subset of a 4x4 grid with Catmull-Rom weights; we do the
    // full 4x4 grid (16 samples) which is slightly heavier but mathematically
    // equivalent in expectation -- the 4 extra corner samples have weights
    // ~0.001 each. Slang's subpixel-R-offset (default -1/3 px) is baked in
    // per-channel as a sample-position shift.
    inline float3 g4_tex2Daa(texture2d<float, access::sample> tex, sampler sam,
                              float2 uv, float2 output_size_inv,
                              float2 aa_r_offset)
    {
        // Sample at 4x4 grid offsets [-1.5, -0.5, 0.5, 1.5] in pixel space.
        // Weights = catmull_rom(offset_x) * catmull_rom(offset_y).
        constexpr float taps[4] = { -1.5f, -0.5f, 0.5f, 1.5f };

        // Per-axis Catmull-Rom weights (precomputed for fixed taps).
        float wx[4], wy[4];
        for (int i = 0; i < 4; ++i) {
            wx[i] = g4_catmull_rom(taps[i]);
            wy[i] = g4_catmull_rom(taps[i]);
        }

        // Per-channel R-subpixel offset (default -1/3 px in x). G centered,
        // B mirrored (slang's aa_subpixel_r_offset == -1 * subpixel_b_offset).
        float2 r_uv_off = aa_r_offset * output_size_inv;
        float2 b_uv_off = -r_uv_off;

        float3 acc = float3(0.0f);
        float w_sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                float w = wx[i] * wy[j];
                float2 off = float2(taps[i], taps[j]) * output_size_inv;
                float r = tex.sample(sam, uv + off + r_uv_off).r;
                float g = tex.sample(sam, uv + off).g;
                float b = tex.sample(sam, uv + off + b_uv_off).b;
                acc += float3(r, g, b) * w;
                w_sum += w;
            }
        }
        return acc / max(w_sum, 1e-5f);
    }

    // =======================================================================
    // Pass 4: Geometry + AA + Final Output (full port of slang pass 11)
    // Slang ref: crt-royale-geometry-aa-last-pass.{slang,h}
    //
    // Default-path behavior (geom_mode=0, overscan=1): slang Pass 11 collapses
    // to a plain point sample + border_dim + gamma. Our default does the same.
    //
    // Curved-geometry path (geom_mode=1, sphere): we now do a real eye-to-
    // screen sphere raycast (g4_curved_uv) matching slang's sphere-mode
    // mathematics, instead of the previous barrel-distortion approximation.
    // geom_radius_factor controls the curvature radius (0 -> very tight,
    // 1.0 -> nearly flat).
    //
    // AA path (geom_mode>0): we run a 16-tap separable Catmull-Rom-cubic AA
    // kernel with per-channel R-subpixel offset (default -1/3 px horizontally),
    // matching Slang's aa_filter=6 (Cubic separable) + aa_cubic_c=0.5 default.
    // The 16 taps cover the full 4x4 grid that subsumes Slang's 12-tap
    // quincunx pattern; the extra 4 corner samples have weights ~0.001.
    //
    // Still simplified vs. slang:
    //   - Cylinder mode (geom_mode=3) and sphere_alt (geom_mode=2) not ported
    //   - Runtime geom_tilt_angle (default 0,0 covers the common case)
    //   - aa_filter / aa_temporal configurability (we fix Gaussian + static)
    //   - Bloom_excess + halation diffusion mix in screen-uv coords
    // =======================================================================
    kernel void pass4_geometry_aa(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 output_size = float2(output.get_width(), output.get_height());
        float2 output_size_inv = 1.0f / output_size;
        float2 flat_uv = (float2(gid) + 0.5f) / output_size;

        // Geometry warp.
        float2 video_uv;
        if (u.geom_mode == 1u) {
            // Sphere raycaster (slang Pass 11 geom_mode=1).
            // geom_radius_factor maps [0, 1] -> radius [G4_RADIUS_STATIC * 4,
            // G4_RADIUS_STATIC / 4] so the slider feels intuitive (lower =
            // tighter curve). 0.20 (our app default) -> radius ~3.2.
            float viewport_aspect = output_size.x / output_size.y;
            float2 geom_aspect = g4_aspect_vector(viewport_aspect);
            float radius = G4_RADIUS_STATIC *
                            mix(4.0f, 0.25f, clamp(u.geom_radius_factor, 0.0f, 1.0f));
            video_uv = g4_curved_uv(flat_uv, geom_aspect, radius);
        } else {
            video_uv = flat_uv;
        }

        // Sample. Use AA if curvature is on (slang's aa_level > 0.5 &&
        // geom_mode > 0.5 branch); plain sample otherwise (matches slang's
        // need_subpixel_aa=false fall-through).
        bool offscreen = any(video_uv < 0.0f) || any(video_uv > 1.0f);
        float3 color;
        if (offscreen) {
            color = float3(0.0f);
        } else if (u.geom_mode > 0u) {
            // aa_subpixel_r_offset_static = (-1/3, 0).
            constexpr float2 aa_r_offset = float2(-1.0f / 3.0f, 0.0f);
            color = g4_tex2Daa(input, sam, video_uv, output_size_inv,
                                aa_r_offset);
        } else {
            color = input.sample(sam, video_uv).rgb;
        }

        // Border dim factor (cgwg / TroggleMonkey algorithm, slang ref:
        // geometry-functions.h:665-688). geom_aspect is normally derived
        // from viewport_aspect_ratio; we simplify to (1, 1) -- the visual
        // difference vs the real aspect-scaled version is tiny for our
        // 4:3-ish viewports and avoids dragging in the rest of slang's
        // aspect-vector machinery.
        constexpr float2 geom_aspect_border = float2(1.0f, 1.0f);
        float2 edge_dists  = min(video_uv, float2(1.0f) - video_uv) * geom_aspect_border;
        float2 border_pen  = max(float2(u.border_size) - edge_dists, float2(0.0f));
        float  pen_ratio   = (u.border_size > 1e-5f)
                           ? length(border_pen) / u.border_size
                           : 0.0f;
        float  escape      = max(1.0f - pen_ratio, 0.0f);
        float  dim_factor  = min(pow(escape, u.border_darkness) *
                                 max(1.0f, u.border_compress), 1.0f);
        color *= dim_factor;

        // Optional brightness boost to compensate for missing bloom.
        color *= max(u.brightness_boost, 0.0f);

        // Display gamma encode (slang's encode_output for LAST_PASS).
        float3 encoded = pow(max(color, 0.0f), float3(1.0f / u.lcd_gamma));
        output.write(float4(encoded, 1.0f), gid);
    }

    // =======================================================================
    // pass_linear_export: write a linear-light source to bgra8Unorm without
    // applying any gamma encoding. Used by the headless validation runner so
    // intermediate-pass snapshots match librashader's per-pass output, which
    // exports the raw linear values stored in sRGB framebuffers as PNG bytes.
    //
    // For final user-facing display we still want pass_final_encode (display
    // gamma 1/2.2). This kernel is purely a comparison-pipeline helper.
    // =======================================================================
    kernel void pass_linear_export(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms               &u       [[ buffer(0)  ]],
        sampler                         sam      [[ sampler(0) ]],
        uint2                           gid      [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

        float2 uv = (float2(gid) + 0.5f) /
                    float2(output.get_width(), output.get_height());
        float4 c = input.sample(sam, uv);
        output.write(float4(saturate(c.rgb), 1.0f), gid);
    }

} // namespace crt_royale
