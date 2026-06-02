// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Pass 4: Geometry + AA + Final Output
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Border-dim algorithm originally by cgwg (also GPL).
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang reference: crt-royale-geometry-aa-last-pass.{slang,h}
// Slang pass index: 11 (our pass 4 corresponds to slang's pass 11).
//
// ==================================================================
// IMPORTANT: this file is the original Iteration-13 study mirror and
// only documents the SIMPLIFIED variant. The full port -- including the
// real Slang sphere raycaster (geometry-functions.h:563-663) and a
// 16-tap separable Catmull-Rom-cubic AA kernel -- is documented in the
// companion study mirror CrtRoyalePass11FullPort.metal (and lives in
// the production CrtRoyale.metal as `pass4_geometry_aa`).
// ==================================================================
//
// Pipeline position in our MSL implementation (post-Iteration-16):
//   ... -> BLOOM_FINAL (slang pass 10 alias) -> pass4_geometry_aa -> output
//
// Slang's pass 11 main() at default settings (geom_mode=0, overscan=1,
// aa_level=12, halation_weight=0, no PHOSPHOR_BLOOM_FAKE) reduces to:
//   1. color = tex2D_linearize(input_texture, tex_uv).rgb
//   2. color *= get_border_dim_factor(video_uv, geom_aspect)
//   3. encode_output(color)              // display gamma
// Our default path implements exactly these three steps -- bit-exact to
// Slang within the limits of float precision.
//
// What this iteration adds (also see CrtRoyalePass11FullPort.metal):
//
//   - Real sphere-raycaster for geom_mode=1 (slang's
//     get_curved_video_uv_coords_and_tangent_matrix path with eye-pos,
//     ray-sphere intersection, sphere_xyz_to_uv great-circle mapping).
//     Replaces the prior `c *= 1 + k * |c|^2` barrel approximation.
//   - 16-tap separable Catmull-Rom-cubic AA (`g4_tex2Daa`) used when
//     curvature is active. Slang's aa_filter=6 + aa_cubic_c=0.5.
//   - Subpixel R offset (-1/3 px) baked into the AA kernel.
//
// Still simplified vs. Slang:
//   - geom_mode 2 (sphere_alt) and 3 (cylinder) not ported.
//   - aa_filter configurability (10 filter types in Slang) fixed.
//   - Runtime geom_tilt_angle not implemented (Slang default 0,0).
//
// NOTE: This file is the standalone study mirror. The kernel that
// gets built lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal
// and is significantly more involved than the simplified version below.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // The full Uniforms struct lives in CrtRoyale.metal. Relevant fields:
    //   uint  geom_mode;             // 0 = flat (slang default), 1 = sphere
    //   float geom_radius_factor;    // controls sphere radius (Iter-16 onward)
    //   float border_size;           // 0.015 slang default
    //   float border_darkness;       // 2.0 slang default
    //   float border_compress;       // 2.5 slang default
    //   float brightness_boost;      // 1.0 = slang-equivalent
    //   float lcd_gamma;             // 2.2 slang default

    kernel void pass4_geometry_aa(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        // 1. Compute flat or curved video_uv.
        //    - geom_mode == 0: video_uv = flat_uv (Slang default fall-through).
        //    - geom_mode == 1: g4_curved_uv(flat_uv, geom_aspect, radius)
        //      (full Slang sphere raycaster -- see CrtRoyalePass11FullPort.metal).
        //
        // 2. Sample input.
        //    - off-screen: return black (bezel).
        //    - geom_mode > 0: g4_tex2Daa (16-tap Catmull-Rom + subpixel R offset).
        //    - else: plain bilinear sample (matches Slang's tex2D_linearize
        //      fall-through at default settings).
        //
        // 3. Border dim factor (cgwg / TroggleMonkey, geometry-functions.h:665).
        //    color *= dim_factor.
        //
        // 4. Optional brightness_boost (legacy compensation; default 1.0).
        //
        // 5. Display gamma encode (Slang's encode_output for LAST_PASS).
    }

} // namespace crt_royale
