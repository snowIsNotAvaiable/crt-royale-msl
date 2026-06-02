// -----------------------------------------------------------------------------
// CRT-Royale for RetroVisor
//
// MSL port of TroggleMonkey's CRT-Royale shader.
// Currently implements: Pass 1 (Linearize + Bob) + Pass 2 (Vertical Scanlines)
// + Pass 3 (Apply Phosphor Mask, simplified) + Final Encode.
// Includes per-pass debug picker and PNG snapshot export.
//
// Pass-numbering note: our Pass 3 corresponds to Slang Pass 7
// (`crt-royale-scanlines-horizontal-apply-mask.slang`). Slang Passes 2..6
// (Bloom Approx, Halation Blur V/H, Mask Resize V/H) are not yet ported and
// our Pass 3 uses a simplified procedural aperture grille + no halation. See
// docs/Status.md for the full mapping.
// -----------------------------------------------------------------------------

import MetalKit
import ImageIO
import UniformTypeIdentifiers

@MainActor
final class CrtRoyale: Shader {

    // Must match the layout in CrtRoyale.metal exactly. Field order matters.
    struct Uniforms {

        // Gamma
        var crt_gamma: Float
        var lcd_gamma: Float

        // Interlacing
        var interlace_detect: UInt32
        var interlace_bff: UInt32
        var interlace_1080i: UInt32
        var frame_count: UInt32

        // Sizes
        var texture_size: SIMD2<Float>
        var video_size: SIMD2<Float>
        var output_size: SIMD2<Float>

        // Beam (pass 2)
        var beam_min_sigma: Float
        var beam_max_sigma: Float
        var beam_spot_power: Float
        var beam_min_shape: Float
        var beam_max_shape: Float
        var beam_shape_power: Float
        var levels_autodim_temp: Float

        // Mask (pass 3)
        var mask_type: UInt32          // 0 = aperture grille (only one supported)
        var mask_triad_size: Float     // output pixels per RGB triad
        var mask_amplify: Float        // brightness comp for mask dimming
        var mask_lut_enabled: UInt32   // 0 = procedural fallback, 1 = sample LUT
        var mask_sample_mode: UInt32   // 0 = sample MASK_RESIZE (slang default,
                                       //     Lanczos via Pass 5+6),
                                       // 1 = hardware-resample (mipmap+aniso)

        // Halation / Diffusion (slang passes 3+4 + apply-mask + bloom-final)
        var halation_weight: Float     // [0, 1], slang default 0.0
        var diffusion_weight: Float    // [0, 1], slang default 0.075

        // Geometry / final encode (pass 4, slang pass 11)
        var geom_mode: UInt32          // 0=flat (slang default), 1=curved barrel
        var geom_radius_factor: Float  // barrel-distortion strength
        var border_size: Float         // [0, 0.5]
        var border_darkness: Float     // [0, inf)
        var border_compress: Float     // [1, inf)
        var brightness_boost: Float    // post-border multiply

        // Debug pipeline
        var debug_pass_index: UInt32   // 0=final, 1=pass1, 2=pass2, 3=pass3

        static let defaults = Uniforms(
            crt_gamma: 2.5,
            lcd_gamma: 2.2,
            interlace_detect: 1,
            interlace_bff: 0,
            interlace_1080i: 0,
            frame_count: 0,
            texture_size: [0, 0],
            video_size: [0, 0],
            output_size: [0, 0],
            beam_min_sigma: 0.02,
            beam_max_sigma: 0.30,
            beam_spot_power: 0.33,
            beam_min_shape: 2.0,
            beam_max_shape: 4.0,
            beam_shape_power: 0.25,
            levels_autodim_temp: 0.5,
            // mask_amplify currently unused; pass3 emits dim product (slang-eq).
            // Compensated for visually via brightness_boost in pass 4.
            mask_type: 0,
            mask_triad_size: 3.0,
            mask_amplify: 4.811320754717,
            mask_lut_enabled: 1,
            mask_sample_mode: 1,         // hardware-resample (validated path).
                                         // mode 0 (sample MASK_RESIZE) is
                                         // implemented but not yet bit-exact
                                         // vs. Slang -- see Status.md.
            halation_weight: 0.0,
            diffusion_weight: 0.075,
            // Pass 4 defaults: barrel curvature ON for app demo (slang default
            // is 0/flat -- we override here because the iconic CRT look is
            // the whole point of the on-screen preview). brightness_boost is
            // now 1.0 because BLOOM_FINAL (the new Pass 4 input) already
            // bakes the un-dim + mask-amplify into its values; legacy hack
            // no longer needed.
            geom_mode: 1,
            geom_radius_factor: 0.20,
            border_size: 0.015,
            border_darkness: 2.0,
            border_compress: 2.5,
            brightness_boost: 1.0,
            debug_pass_index: 0
        )
    }

    var uniforms: Uniforms = .defaults
    var frameCounter: UInt32 = 0

    // Compute kernels
    var linearizeKernel: Kernel!
    var pass2Kernel: Kernel!
    var pass3Kernel: Pass3Kernel!
    var pass4Kernel: Kernel!
    var bloomApproxKernel: Kernel!
    var halationVKernel: Kernel!
    var halationHKernel: Kernel!
    var maskResizeVKernel: Kernel!
    var maskResizeHKernel: Kernel!
    var brightpassKernel: BrightpassKernel!
    var bloomVKernel: Kernel!
    var bloomHKernel: BloomHReconstituteKernel!
    var finalEncodeKernel: Kernel!

    // Intermediate textures (linear-light, rgba16Float for HDR headroom)
    var linearized: MTLTexture!
    var scanlinesVertical: MTLTexture!
    var maskedScanlines: MTLTexture!
    var bloomApprox: MTLTexture!
    var halationV: MTLTexture!
    var halationBlur: MTLTexture!
    var maskResizeV: MTLTexture!
    var maskResize: MTLTexture!         // alias MASK_RESIZE
    var brightpass: MTLTexture!
    var bloomV: MTLTexture!
    var bloomFinal: MTLTexture!

    // Phosphor-mask LUT (loaded once, mipmapped). Lives in the bundle next to
    // the shader sources. nil if the file isn't shipped -- pass3 falls back
    // to the procedural aperture grille in that case.
    var maskLut: MTLTexture?           // Currently-active large LUT (hardware-resample, mode=1)
    var maskLutGrille: MTLTexture?     // mask_type = 0
    var maskLutSlot:   MTLTexture?     // mask_type = 1
    var maskLutShadow: MTLTexture?     // mask_type = 2
    // Small LUTs (64x64, no mipmap) consumed by Mask-Resize V (slang Pass 5).
    // Slang uses these when PHOSPHOR_MASK_RESIZE_MIPMAPPED_LUT is undefined
    // (the Metal default on librashader).
    var maskLutGrilleSmall: MTLTexture?
    var maskLutSlotSmall:   MTLTexture?
    var maskLutShadowSmall: MTLTexture?
    var lutSampler: MTLSamplerState?

    // Snapshot trigger (edge-detected via the "Save Snapshot" setting).
    private var snapshotPending: Bool = false

    init() {

        super.init(name: "CRT-Royale")

        settings = [

            Group(title: "Gamma", [

                ShaderSetting(
                    title: "CRT Gamma",
                    range: 1.0...5.0, step: 0.05,
                    value: Binding(
                        key: "CRT_GAMMA",
                        get: { [unowned self] in self.uniforms.crt_gamma },
                        set: { [unowned self] in self.uniforms.crt_gamma = $0 })),

                ShaderSetting(
                    title: "LCD Gamma",
                    range: 1.0...5.0, step: 0.05,
                    value: Binding(
                        key: "LCD_GAMMA",
                        get: { [unowned self] in self.uniforms.lcd_gamma },
                        set: { [unowned self] in self.uniforms.lcd_gamma = $0 })),
            ]),

            Group(title: "Beam", [

                ShaderSetting(
                    title: "Min Sigma",
                    range: 0.0...0.5, step: 0.005,
                    value: Binding(
                        key: "BEAM_MIN_SIGMA",
                        get: { [unowned self] in self.uniforms.beam_min_sigma },
                        set: { [unowned self] in self.uniforms.beam_min_sigma = $0 })),

                ShaderSetting(
                    title: "Max Sigma",
                    range: 0.0...1.0, step: 0.005,
                    value: Binding(
                        key: "BEAM_MAX_SIGMA",
                        get: { [unowned self] in self.uniforms.beam_max_sigma },
                        set: { [unowned self] in self.uniforms.beam_max_sigma = $0 })),

                ShaderSetting(
                    title: "Spot Power",
                    range: 0.0...16.0, step: 0.05,
                    value: Binding(
                        key: "BEAM_SPOT_POWER",
                        get: { [unowned self] in self.uniforms.beam_spot_power },
                        set: { [unowned self] in self.uniforms.beam_spot_power = $0 })),

                ShaderSetting(
                    title: "Min Shape",
                    range: 2.0...32.0, step: 0.1,
                    value: Binding(
                        key: "BEAM_MIN_SHAPE",
                        get: { [unowned self] in self.uniforms.beam_min_shape },
                        set: { [unowned self] in self.uniforms.beam_min_shape = $0 })),

                ShaderSetting(
                    title: "Max Shape",
                    range: 2.0...32.0, step: 0.1,
                    value: Binding(
                        key: "BEAM_MAX_SHAPE",
                        get: { [unowned self] in self.uniforms.beam_max_shape },
                        set: { [unowned self] in self.uniforms.beam_max_shape = $0 })),

                ShaderSetting(
                    title: "Shape Power",
                    range: 0.0...16.0, step: 0.05,
                    value: Binding(
                        key: "BEAM_SHAPE_POWER",
                        get: { [unowned self] in self.uniforms.beam_shape_power },
                        set: { [unowned self] in self.uniforms.beam_shape_power = $0 })),
            ]),

            Group(title: "Mask", [

                ShaderSetting(
                    title: "Mask Type",
                    items: [
                        ("Aperture Grille", 0),
                        ("Slot Mask",        1),
                        ("Shadow Mask EDP",  2),
                    ],
                    value: Binding(
                        key: "MASK_TYPE",
                        get: { [unowned self] in Float(self.uniforms.mask_type) },
                        set: { [unowned self] in self.uniforms.mask_type = UInt32($0) })),

                ShaderSetting(
                    title: "Triad Size (px)",
                    range: 1.0...18.0, step: 0.5,
                    value: Binding(
                        key: "MASK_TRIAD_SIZE",
                        get: { [unowned self] in self.uniforms.mask_triad_size },
                        set: { [unowned self] in self.uniforms.mask_triad_size = $0 })),

                // mask_amplify is no longer user-tunable here -- it is set
                // automatically in apply() based on mask_type (using the
                // slang-derived 1/mask_*_avg_color constants). Adding the
                // slider back would require also disabling the auto-update.

                ShaderSetting(
                    title: "Sample Mode",
                    items: [
                        ("MASK_RESIZE (Lanczos)", 0),
                        ("Hardware (mipmap+aniso)", 1),
                    ],
                    value: Binding(
                        key: "MASK_SAMPLE_MODE",
                        get: { [unowned self] in Float(self.uniforms.mask_sample_mode) },
                        set: { [unowned self] in self.uniforms.mask_sample_mode = UInt32($0) })),
            ]),

            Group(title: "Halation", [

                ShaderSetting(
                    title: "Halation Weight",
                    range: 0.0...1.0, step: 0.005,
                    value: Binding(
                        key: "HALATION_WEIGHT",
                        get: { [unowned self] in self.uniforms.halation_weight },
                        set: { [unowned self] in self.uniforms.halation_weight = $0 })),

                ShaderSetting(
                    title: "Diffusion Weight",
                    range: 0.0...1.0, step: 0.005,
                    value: Binding(
                        key: "DIFFUSION_WEIGHT",
                        get: { [unowned self] in self.uniforms.diffusion_weight },
                        set: { [unowned self] in self.uniforms.diffusion_weight = $0 })),
            ]),

            Group(title: "Geometry & Border", [

                ShaderSetting(
                    title: "Geometry Mode",
                    items: [
                        ("Flat (Slang default)", 0),
                        ("Curved (barrel approximation)", 1),
                    ],
                    value: Binding(
                        key: "GEOM_MODE",
                        get: { [unowned self] in Float(self.uniforms.geom_mode) },
                        set: { [unowned self] in self.uniforms.geom_mode = UInt32($0) })),

                ShaderSetting(
                    title: "Curvature Strength",
                    range: 0.0...0.6, step: 0.01,
                    value: Binding(
                        key: "GEOM_RADIUS_FACTOR",
                        get: { [unowned self] in self.uniforms.geom_radius_factor },
                        set: { [unowned self] in self.uniforms.geom_radius_factor = $0 })),

                ShaderSetting(
                    title: "Border Size",
                    range: 0.0...0.1, step: 0.005,
                    value: Binding(
                        key: "BORDER_SIZE",
                        get: { [unowned self] in self.uniforms.border_size },
                        set: { [unowned self] in self.uniforms.border_size = $0 })),

                ShaderSetting(
                    title: "Border Darkness",
                    range: 0.0...8.0, step: 0.1,
                    value: Binding(
                        key: "BORDER_DARKNESS",
                        get: { [unowned self] in self.uniforms.border_darkness },
                        set: { [unowned self] in self.uniforms.border_darkness = $0 })),

                ShaderSetting(
                    title: "Brightness Boost",
                    range: 1.0...20.0, step: 0.25,
                    value: Binding(
                        key: "BRIGHTNESS_BOOST",
                        get: { [unowned self] in self.uniforms.brightness_boost },
                        set: { [unowned self] in self.uniforms.brightness_boost = $0 })),
            ]),

            Group(title: "Interlacing", [

                ShaderSetting(
                    title: "Detect Interlacing",
                    range: nil, step: 1.0,
                    value: Binding(
                        key: "INTERLACE_DETECT",
                        get: { [unowned self] in Float(self.uniforms.interlace_detect) },
                        set: { [unowned self] in self.uniforms.interlace_detect = UInt32($0) })),

                ShaderSetting(
                    title: "Bottom Field First",
                    range: nil, step: 1.0,
                    value: Binding(
                        key: "INTERLACE_BFF",
                        get: { [unowned self] in Float(self.uniforms.interlace_bff) },
                        set: { [unowned self] in self.uniforms.interlace_bff = UInt32($0) })),

                ShaderSetting(
                    title: "Detect 1080i",
                    range: nil, step: 1.0,
                    value: Binding(
                        key: "INTERLACE_1080I",
                        get: { [unowned self] in Float(self.uniforms.interlace_1080i) },
                        set: { [unowned self] in self.uniforms.interlace_1080i = UInt32($0) })),
            ]),

            Group(title: "Debug", [

                ShaderSetting(
                    title: "Show Pass",
                    items: [
                        ("Final", 0),
                        ("Pass 1 (Linearize)", 1),
                        ("Pass 2 (Scanlines V)", 2),
                        ("Pass 3 (Apply Mask)", 3),
                    ],
                    value: Binding(
                        key: "DEBUG_PASS_INDEX",
                        get: { [unowned self] in Float(self.uniforms.debug_pass_index) },
                        set: { [unowned self] in self.uniforms.debug_pass_index = UInt32($0) })),

                // Edge-triggered snapshot button: any non-zero set arms a
                // capture, which the next apply() consumes and resets to 0.
                ShaderSetting(
                    title: "Save Snapshot",
                    range: 0.0...1.0, step: 1.0,
                    value: Binding(
                        key: "SAVE_SNAPSHOT",
                        get: { 0 },
                        set: { [unowned self] v in
                            if v != 0 { self.snapshotPending = true }
                        })),
            ]),
        ]
    }

    override var presets: [String] { ["Default", "Flat Panel"] }

    override func revertToPreset(nr: Int) {

        switch nr {
        case 1:
            // "Flat Panel" preset: minimal CRT character, mostly identity.
            uniforms = .defaults
            uniforms.crt_gamma = 2.2
            uniforms.lcd_gamma = 2.2
            uniforms.beam_max_sigma = 0.5    // wide, soft beam = no scanlines
        default:
            uniforms = .defaults
        }
    }

    override func activate() {

        super.activate()

        linearizeKernel   = Pass1Kernel(sampler: ShaderLibrary.linear)
        pass2Kernel       = Pass2Kernel(sampler: ShaderLibrary.linear)
        pass3Kernel       = Pass3Kernel(sampler: ShaderLibrary.linear)
        pass4Kernel       = Pass4Kernel(sampler: ShaderLibrary.linear)
        bloomApproxKernel = BloomApproxKernel(sampler: ShaderLibrary.linear)
        halationVKernel   = HalationVKernel(sampler: ShaderLibrary.linear)
        halationHKernel   = HalationHKernel(sampler: ShaderLibrary.linear)
        maskResizeVKernel = MaskResizeVKernel(sampler: ShaderLibrary.linear)
        maskResizeHKernel = MaskResizeHKernel(sampler: ShaderLibrary.linear)
        brightpassKernel  = BrightpassKernel(sampler: ShaderLibrary.linear)
        bloomVKernel      = BloomVKernel(sampler: ShaderLibrary.linear)
        bloomHKernel      = BloomHReconstituteKernel(sampler: ShaderLibrary.linear)
        finalEncodeKernel = FinalEncodeKernel(sampler: ShaderLibrary.linear)

        // Linear + repeat + mipmap = match slang's mask_grille_texture_large
        // sampler settings (_linear=true, _wrap_mode=repeat, _mipmap=true).
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.mipFilter = .linear
        sd.sAddressMode = .repeat
        sd.tAddressMode = .repeat
        sd.maxAnisotropy = 16
        lutSampler = ShaderLibrary.device.makeSamplerState(descriptor: sd)

        // Load all three mask LUTs from the app bundle so the user can switch
        // between aperture grille, slot, and shadow masks at runtime via the
        // Mask Type picker. Any that fail to load fall back to procedural.
        maskLutGrille = Self.loadMaskLUT(named: "TileableLinearApertureGrille15Wide8And5d5Spacing")
        maskLutSlot   = Self.loadMaskLUT(named: "TileableLinearSlotMaskTall15Wide9And4d5Horizontal9d14VerticalSpacing")
        maskLutShadow = Self.loadMaskLUT(named: "TileableLinearShadowMaskEDP")
        maskLut = maskLutGrille
        if maskLutGrille == nil { uniforms.mask_lut_enabled = 0 }
        // Small LUTs for slang's Pass 5 input (PHOSPHOR_MASK_RESIZE_MIPMAPPED_LUT
        // undefined branch). 64x64, no mipmap.
        maskLutGrilleSmall = Self.loadMaskLUT(
            named: "TileableLinearApertureGrille15Wide8And5d5SpacingResizeTo64",
            mipmap: false)
        maskLutSlotSmall = Self.loadMaskLUT(
            named: "TileableLinearSlotMaskTall15Wide9And4d5Horizontal9d14VerticalSpacingResizeTo64",
            mipmap: false)
        maskLutShadowSmall = Self.loadMaskLUT(
            named: "TileableLinearShadowMaskEDPResizeTo64",
            mipmap: false)
    }

    // Average-color constants per mask type. Slang ref:
    // user-cgp-constants.h:40,43,46. amplify = 1.0 / avg_color.
    static let maskAmplifyByType: [UInt32: Float] = [
        0: 255.0 / 53.0,  // grille15 -> ~4.811
        1: 255.0 / 46.0,  // slot     -> ~5.543
        2: 255.0 / 41.0,  // shadowEDP-> ~6.220
    ]

    // Loads a "TileableLinear*.png" mask texture from the main bundle into a
    // mipmapped MTLTexture suitable for slang-equivalent mask sampling.
    // The PNG bytes are interpreted as already-linear (no sRGB decode) and
    // the mipmap chain is generated on the GPU.
    static func loadMaskLUT(named name: String, mipmap: Bool = true) -> MTLTexture? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "png") else {
            return nil
        }
        let loader = MTKTextureLoader(device: ShaderLibrary.device)
        let opts: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: mipmap,
            // The LUT bytes are linear; tagging the texture as non-sRGB stops
            // the sampler from inserting a gamma decode.
            .SRGB: false,
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue),
        ]
        return try? loader.newTexture(URL: url, options: opts)
    }

    func updateTextures(in input: MTLTexture, out output: MTLTexture) {

        let w = output.width
        let h = output.height

        // Use rgba16Float for intermediates: beam contributions can exceed
        // 1.0 before levels_autodim_temp clamps them, so 8-bit unorm would
        // silently clip.
        if linearized?.width != w || linearized?.height != h {
            linearized = Shader.makeTexture("crt-royale-linear",
                                            width: w, height: h,
                                            pixelFormat: .rgba16Float)
        }
        if scanlinesVertical?.width != w || scanlinesVertical?.height != h {
            scanlinesVertical = Shader.makeTexture("crt-royale-scanlines-v",
                                                   width: w, height: h,
                                                   pixelFormat: .rgba16Float)
        }
        if maskedScanlines?.width != w || maskedScanlines?.height != h {
            maskedScanlines = Shader.makeTexture("crt-royale-masked-scanlines",
                                                 width: w, height: h,
                                                 pixelFormat: .rgba16Float)
        }
        // BLOOM_APPROX: slang pass 2 absolute scale 320x240. Reallocate only
        // if it hasn't been created yet -- the size is fixed.
        if bloomApprox == nil {
            bloomApprox = Shader.makeTexture("crt-royale-bloom-approx",
                                             width: 320, height: 240,
                                             pixelFormat: .rgba16Float)
        }
        // HALATION_V + HALATION_BLUR: slang passes 3+4 inherit BLOOM_APPROX
        // size via scale_type=source, scale=1.0. Fixed at 320x240.
        if halationV == nil {
            halationV = Shader.makeTexture("crt-royale-halation-v",
                                           width: 320, height: 240,
                                           pixelFormat: .rgba16Float)
        }
        if halationBlur == nil {
            halationBlur = Shader.makeTexture("crt-royale-halation-blur",
                                              width: 320, height: 240,
                                              pixelFormat: .rgba16Float)
        }
        // MASK_RESIZE_V (slang Pass 5) and MASK_RESIZE (slang Pass 6) sizes
        // depend on the viewport: y is viewport_y * mask_resize_viewport_scale.y
        // (= 0.0625) for both passes, x is absolute 64 for Pass 5 and
        // viewport_x * 0.0625 for Pass 6. Clamped to >=2 to avoid zero-sized
        // textures for tiny viewports.
        let mrY = max(Int((Float(h) * 0.0625).rounded()), 2)
        let mrXh = max(Int((Float(w) * 0.0625).rounded()), 2)
        if maskResizeV?.width != 64 || maskResizeV?.height != mrY {
            maskResizeV = Shader.makeTexture("crt-royale-mask-resize-v",
                                             width: 64, height: mrY,
                                             pixelFormat: .rgba16Float)
        }
        if maskResize?.width != mrXh || maskResize?.height != mrY {
            maskResize = Shader.makeTexture("crt-royale-mask-resize",
                                            width: mrXh, height: mrY,
                                            pixelFormat: .rgba16Float)
        }
        // BRIGHTPASS: slang pass 8 at viewport scale (= masked-scanlines size).
        if brightpass?.width != w || brightpass?.height != h {
            brightpass = Shader.makeTexture("crt-royale-brightpass",
                                            width: w, height: h,
                                            pixelFormat: .rgba16Float)
        }
        // BLOOM_V: slang pass 9, same viewport scale.
        if bloomV?.width != w || bloomV?.height != h {
            bloomV = Shader.makeTexture("crt-royale-bloom-v",
                                        width: w, height: h,
                                        pixelFormat: .rgba16Float)
        }
        // BLOOM_FINAL: slang pass 10, same viewport scale.
        if bloomFinal?.width != w || bloomFinal?.height != h {
            bloomFinal = Shader.makeTexture("crt-royale-bloom-final",
                                            width: w, height: h,
                                            pixelFormat: .rgba16Float)
        }
    }

    override func apply(commandBuffer: MTLCommandBuffer,
                        in input: MTLTexture, out output: MTLTexture, rect: CGRect) {

        updateTextures(in: input, out: output)

        // Per-frame uniforms
        frameCounter &+= 1
        uniforms.frame_count  = frameCounter
        uniforms.texture_size = SIMD2<Float>(Float(input.width), Float(input.height))
        uniforms.video_size   = SIMD2<Float>(Float(input.width), Float(input.height))
        uniforms.output_size  = SIMD2<Float>(Float(output.width), Float(output.height))

        // Pass 1: Linearize CRT gamma + bob interlaced fields
        linearizeKernel.apply(commandBuffer: commandBuffer,
                              source: input, target: linearized,
                              options: &uniforms,
                              length: MemoryLayout<Uniforms>.stride)

        // Pass 2: Vertical scanlines / beam distribution
        pass2Kernel.apply(commandBuffer: commandBuffer,
                          source: linearized, target: scanlinesVertical,
                          options: &uniforms,
                          length: MemoryLayout<Uniforms>.stride)

        // BLOOM_APPROX (slang pass 2): downscaled low-pass version of the
        // linearized source. Consumed by brightpass + halation V/H.
        bloomApproxKernel.apply(commandBuffer: commandBuffer,
                                source: linearized, target: bloomApprox,
                                options: &uniforms,
                                length: MemoryLayout<Uniforms>.stride)

        // HALATION_V (slang pass 3): vertical 9-tap Gauss of BLOOM_APPROX.
        halationVKernel.apply(commandBuffer: commandBuffer,
                              source: bloomApprox, target: halationV,
                              options: &uniforms,
                              length: MemoryLayout<Uniforms>.stride)

        // HALATION_BLUR (slang pass 4): horizontal 9-tap Gauss of halationV.
        // Sampled by pass3 (apply-mask, halation_weight) and pass 10
        // (bloom_h_reconstitute, diffusion_weight).
        halationHKernel.apply(commandBuffer: commandBuffer,
                              source: halationV, target: halationBlur,
                              options: &uniforms,
                              length: MemoryLayout<Uniforms>.stride)

        // Pick the LUT + mask_amplify matching the user-selected mask type.
        // Falls back to grille (mask_type=0) if the requested PNG didn't
        // ship in the bundle.
        let activeLut: MTLTexture?
        let activeLutSmall: MTLTexture?
        switch uniforms.mask_type {
        case 1:
            activeLut      = maskLutSlot      ?? maskLutGrille
            activeLutSmall = maskLutSlotSmall ?? maskLutGrilleSmall
        case 2:
            activeLut      = maskLutShadow      ?? maskLutGrille
            activeLutSmall = maskLutShadowSmall ?? maskLutGrilleSmall
        default:
            activeLut      = maskLutGrille
            activeLutSmall = maskLutGrilleSmall
        }
        uniforms.mask_amplify =
            Self.maskAmplifyByType[uniforms.mask_type] ?? (255.0 / 53.0)

        // MASK_RESIZE_V (slang Pass 5) + MASK_RESIZE (slang Pass 6). Both
        // discard outside MR_NUM_TILES tiles, so only the first 2x2 tile area
        // gets real Lanczos-resampled data; the rest is filled with zeros.
        // This is fine -- Pass 7 only reads the first tile region.
        if let lutSmall = activeLutSmall {
            maskResizeVKernel.apply(commandBuffer: commandBuffer,
                                    source: lutSmall, target: maskResizeV,
                                    options: &uniforms,
                                    length: MemoryLayout<Uniforms>.stride)
            maskResizeHKernel.apply(commandBuffer: commandBuffer,
                                    source: maskResizeV, target: maskResize,
                                    options: &uniforms,
                                    length: MemoryLayout<Uniforms>.stride)
        }

        // Pass 3: Apply phosphor mask. LUT path (slang-equivalent) when the
        // PNG was loaded; procedural aperture-grille fallback otherwise.
        // mask_sample_mode=0 -> sample MASK_RESIZE; mode=1 -> hardware-resample
        // the large LUT directly. Always samples HALATION_BLUR for the
        // halation_weight lerp.
        pass3Kernel.applyWithLUT(commandBuffer: commandBuffer,
                                 source: scanlinesVertical, target: maskedScanlines,
                                 lut: activeLut, lutSampler: lutSampler,
                                 halationBlur: halationBlur,
                                 maskResize: maskResize,
                                 options: &uniforms,
                                 length: MemoryLayout<Uniforms>.stride)

        // BRIGHTPASS (slang pass 8): area-based brightness extraction.
        // Samples MASKED_SCANLINES + BLOOM_APPROX.
        brightpassKernel.applyWithBloomApprox(commandBuffer: commandBuffer,
                                              source: maskedScanlines,
                                              target: brightpass,
                                              bloomApprox: bloomApprox,
                                              options: &uniforms,
                                              length: MemoryLayout<Uniforms>.stride)

        // BLOOM_V (slang pass 9): vertical 9-tap Gaussian blur of brightpass.
        bloomVKernel.apply(commandBuffer: commandBuffer,
                           source: brightpass, target: bloomV,
                           options: &uniforms,
                           length: MemoryLayout<Uniforms>.stride)

        // BLOOM_FINAL (slang pass 10): horizontal blur of bloomV + add the
        // mask-dim portion of MASKED_SCANLINES back, un-dim, mask-amplify.
        // Pass 4 samples this instead of MASKED_SCANLINES.
        bloomHKernel.applyWithSources(commandBuffer: commandBuffer,
                                      source: bloomV, target: bloomFinal,
                                      brightpass: brightpass,
                                      maskedScanlines: maskedScanlines,
                                      halationBlur: halationBlur,
                                      options: &uniforms,
                                      length: MemoryLayout<Uniforms>.stride)

        // Final routing.
        //  - case 0 (Final): run the real pass 4 (geometry/border/encode).
        //  - case 1/2/3 (debug intermediates): plain gamma encode of the
        //    linear-light texture; bypasses geometry + border so the user
        //    sees what the pass actually produced.
        if uniforms.debug_pass_index == 0 {
            pass4Kernel.apply(commandBuffer: commandBuffer,
                              source: bloomFinal, target: output,
                              options: &uniforms,
                              length: MemoryLayout<Uniforms>.stride)
        } else {
            let displaySource: MTLTexture
            switch uniforms.debug_pass_index {
            case 1:  displaySource = linearized
            case 2:  displaySource = scanlinesVertical
            case 3:  displaySource = maskedScanlines
            default: displaySource = maskedScanlines
            }
            finalEncodeKernel.apply(commandBuffer: commandBuffer,
                                    source: displaySource, target: output,
                                    options: &uniforms,
                                    length: MemoryLayout<Uniforms>.stride)
        }

        // PNG snapshot of the visible (gamma-encoded) output.
        if snapshotPending {
            snapshotPending = false
            scheduleSnapshot(commandBuffer: commandBuffer,
                             texture: output,
                             passIndex: Int(uniforms.debug_pass_index))
        }
    }
}

// MARK: - Kernel definitions

extension CrtRoyale {

    class Pass1Kernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass1_linearize", sampler: sampler)
        }
    }

    class Pass2Kernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass2_vertical_scanlines", sampler: sampler)
        }
    }

    class Pass4Kernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass4_geometry_aa", sampler: sampler)
        }
    }

    class BloomApproxKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_bloom_approx", sampler: sampler)
        }
    }

    class Pass3Kernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass3_apply_mask", sampler: sampler)
        }

        // pass3_apply_mask takes:
        //   texture(0) = source (VERTICAL_SCANLINES)
        //   texture(1) = target (MASKED_SCANLINES)
        //   texture(2) = mask LUT (optional, mask_lut_enabled=1 path)
        //   texture(3) = HALATION_BLUR (slang pass 4 output)
        //   texture(4) = MASK_RESIZE (slang pass 6 output, mask_sample_mode=0)
        //   sampler(0) = standard linear
        //   sampler(1) = LUT sampler (linear+repeat+mipmap)
        // The base class's two-argument apply() can't express all of this,
        // so we duplicate its dispatch loop here.
        func applyWithLUT(commandBuffer: MTLCommandBuffer,
                          source: MTLTexture, target: MTLTexture,
                          lut: MTLTexture?, lutSampler: MTLSamplerState?,
                          halationBlur: MTLTexture,
                          maskResize: MTLTexture?,
                          options: UnsafeRawPointer, length: Int)
        {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setTexture(source, index: 0)
            encoder.setTexture(target, index: 1)
            if let lut = lut { encoder.setTexture(lut, index: 2) }
            encoder.setTexture(halationBlur, index: 3)
            if let mr = maskResize { encoder.setTexture(mr, index: 4) }
            encoder.setSamplerState(sampler ?? ShaderLibrary.linear, index: 0)
            if let ls = lutSampler { encoder.setSamplerState(ls, index: 1) }
            encoder.setComputePipelineState(kernel)
            encoder.setBytes(options, length: length, index: 0)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let groups = MTLSize(
                width: (target.width + tg.width - 1) / tg.width,
                height: (target.height + tg.height - 1) / tg.height,
                depth: 1)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// HalationV / HalationH are plain 1-input separable 9-tap Gaussian
    /// blurs at the BLOOM_APPROX scale (320x240). Slang passes 3+4.
    class HalationVKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_halation_v", sampler: sampler)
        }
    }
    class HalationHKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_halation_h", sampler: sampler)
        }
    }

    /// MaskResizeV / MaskResizeH are single-axis Lanczos-sinc resamplers
    /// for the phosphor-mask LUT. Slang passes 5+6. Standard 1-input
    /// kernels -- the base Kernel.apply() works directly.
    class MaskResizeVKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_mask_resize_v", sampler: sampler)
        }
    }
    class MaskResizeHKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_mask_resize_h", sampler: sampler)
        }
    }

    class FinalEncodeKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_final_encode", sampler: sampler)
        }
    }

    /// Bloom V (slang pass 9) is a plain 1-input separable Gaussian blur,
    /// so the base Kernel.apply() works directly. Subclass exists only to
    /// bind the right kernel function name.
    class BloomVKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_bloom_v", sampler: sampler)
        }
    }

    /// Bloom H + Reconstitute (slang pass 10) needs four input textures:
    ///   texture(0) = bloomV  (vertically-blurred brightpass)
    ///   texture(1) = output  (BLOOM_FINAL)
    ///   texture(2) = brightpass (slang pass 8 output, raw)
    ///   texture(3) = masked_scanlines (our pass 3 output)
    ///   texture(4) = halation_blur (slang pass 4 output)
    class BloomHReconstituteKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_bloom_h_reconstitute",
                      sampler: sampler)
        }

        func applyWithSources(commandBuffer: MTLCommandBuffer,
                              source: MTLTexture, target: MTLTexture,
                              brightpass: MTLTexture,
                              maskedScanlines: MTLTexture,
                              halationBlur: MTLTexture,
                              options: UnsafeRawPointer, length: Int)
        {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setTexture(source,           index: 0)
            encoder.setTexture(target,           index: 1)
            encoder.setTexture(brightpass,       index: 2)
            encoder.setTexture(maskedScanlines,  index: 3)
            encoder.setTexture(halationBlur,     index: 4)
            encoder.setSamplerState(sampler ?? ShaderLibrary.linear, index: 0)
            encoder.setComputePipelineState(kernel)
            encoder.setBytes(options, length: length, index: 0)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let groups = MTLSize(
                width: (target.width + tg.width - 1) / tg.width,
                height: (target.height + tg.height - 1) / tg.height,
                depth: 1)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Brightpass kernel reads MASKED_SCANLINES (texture 0) + writes
    /// BRIGHTPASS (texture 1) + samples BLOOM_APPROX (texture 2). The
    /// extra-input pattern mirrors Pass3Kernel.applyWithLUT.
    class BrightpassKernel: Kernel {
        convenience init?(sampler: MTLSamplerState) {
            self.init(name: "crt_royale::pass_brightpass", sampler: sampler)
        }

        func applyWithBloomApprox(commandBuffer: MTLCommandBuffer,
                                  source: MTLTexture, target: MTLTexture,
                                  bloomApprox: MTLTexture,
                                  options: UnsafeRawPointer, length: Int)
        {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setTexture(source, index: 0)
            encoder.setTexture(target, index: 1)
            encoder.setTexture(bloomApprox, index: 2)
            encoder.setSamplerState(sampler ?? ShaderLibrary.linear, index: 0)
            encoder.setComputePipelineState(kernel)
            encoder.setBytes(options, length: length, index: 0)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let groups = MTLSize(
                width: (target.width + tg.width - 1) / tg.width,
                height: (target.height + tg.height - 1) / tg.height,
                depth: 1)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

// MARK: - PNG snapshot export

extension CrtRoyale {

    /// Schedules a PNG dump of `texture` once the current command buffer
    /// completes. The texture is expected to be a display-encoded (8-bit)
    /// RGBA/BGRA. Files land in:
    ///   ~/Pictures/RetroVisor/CRT-Royale/pass<N>-<yyyyMMdd-HHmmss>.png
    ///
    /// If `texture` is `.private` storage we cannot getBytes() it directly,
    /// so we blit it into a shared staging texture in the same command buffer.
    func scheduleSnapshot(commandBuffer: MTLCommandBuffer,
                          texture: MTLTexture,
                          passIndex: Int)
    {
        let device = ShaderLibrary.device
        let needsStaging = texture.storageMode == .private

        var sourceForReadback: MTLTexture = texture

        if needsStaging {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: texture.pixelFormat,
                width: texture.width, height: texture.height, mipmapped: false)
            desc.storageMode = .shared
            desc.usage = [.shaderRead]
            guard let staging = device.makeTexture(descriptor: desc),
                  let blit    = commandBuffer.makeBlitCommandEncoder()
            else { return }
            blit.copy(from: texture,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOriginMake(0, 0, 0),
                      sourceSize: MTLSizeMake(texture.width, texture.height, 1),
                      to: staging,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOriginMake(0, 0, 0))
            blit.endEncoding()
            sourceForReadback = staging
        } else if texture.storageMode == .managed {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.synchronize(resource: texture)
                blit.endEncoding()
            }
        }

        let target = sourceForReadback
        commandBuffer.addCompletedHandler { _ in
            DispatchQueue.global(qos: .utility).async {
                CrtRoyale.writeTextureAsPNG(target, passIndex: passIndex)
            }
        }
    }

    nonisolated static func writeTextureAsPNG(_ texture: MTLTexture, passIndex: Int) {

        let width  = texture.width
        let height = texture.height
        let bytesPerPixel = 4
        let bytesPerRow   = width * bytesPerPixel
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)

        let region = MTLRegionMake2D(0, 0, width, height)
        bytes.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!,
                             bytesPerRow: bytesPerRow,
                             from: region,
                             mipmapLevel: 0)
        }

        // Most RetroVisor outputs are bgra8Unorm; swap to RGBA for PNG.
        if texture.pixelFormat == .bgra8Unorm ||
           texture.pixelFormat == .bgra8Unorm_srgb
        {
            for i in stride(from: 0, to: bytes.count, by: 4) {
                bytes.swapAt(i, i + 2)
            }
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage  = CGImage(width: width, height: height,
                                     bitsPerComponent: 8, bitsPerPixel: 32,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo,
                                     provider: provider,
                                     decode: nil,
                                     shouldInterpolate: false,
                                     intent: .defaultIntent)
        else { return }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = formatter.string(from: Date())
        let baseDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Pictures/RetroVisor/CRT-Royale", isDirectory: true)
        try? FileManager.default.createDirectory(at: baseDir,
                                                 withIntermediateDirectories: true)
        let url = baseDir.appendingPathComponent("pass\(passIndex)-\(stamp).png")

        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
    }
}
