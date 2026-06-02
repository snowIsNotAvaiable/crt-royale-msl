// -----------------------------------------------------------------------------
// SwiftRunner -- headless validation harness for the CRT-Royale MSL port.
//
// Loads a PNG, runtime-compiles vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal,
// runs the full pipeline (Pass 1 -> Pass 2 -> Final Encode) and dumps a PNG
// per stage so we can validate visually + statistically without launching the
// RetroVisor GUI.
//
// Usage:
//   SwiftRunner --metal <path/to/CrtRoyale.metal> \
//               --input  <input.png> \
//               --outdir <dir-for-snapshots>
// -----------------------------------------------------------------------------

import Foundation
import Metal
import MetalKit
import AppKit
import ImageIO
import UniformTypeIdentifiers

// MARK: - Uniforms (mirrors CrtRoyale.swift / CrtRoyale.metal)

struct Uniforms {
    var crt_gamma: Float
    var lcd_gamma: Float
    var interlace_detect: UInt32
    var interlace_bff: UInt32
    var interlace_1080i: UInt32
    var frame_count: UInt32
    var texture_size: SIMD2<Float>
    var video_size: SIMD2<Float>
    var output_size: SIMD2<Float>
    var beam_min_sigma: Float
    var beam_max_sigma: Float
    var beam_spot_power: Float
    var beam_min_shape: Float
    var beam_max_shape: Float
    var beam_shape_power: Float
    var levels_autodim_temp: Float
    var mask_type: UInt32
    var mask_triad_size: Float
    var mask_amplify: Float
    var mask_lut_enabled: UInt32
    var mask_sample_mode: UInt32
    // Halation / Diffusion (slang passes 3+4 + apply-mask + bloom-final)
    var halation_weight: Float
    var diffusion_weight: Float
    // Pass 4 (slang pass 11) -- geometry + border + final encode
    var geom_mode: UInt32
    var geom_radius_factor: Float
    var border_size: Float
    var border_darkness: Float
    var border_compress: Float
    var brightness_boost: Float
    var debug_pass_index: UInt32

    static let defaults = Uniforms(
        crt_gamma: 2.5, lcd_gamma: 2.2,
        interlace_detect: 1, interlace_bff: 0, interlace_1080i: 0,
        frame_count: 1,
        texture_size: [0, 0], video_size: [0, 0], output_size: [0, 0],
        beam_min_sigma: 0.02, beam_max_sigma: 0.30, beam_spot_power: 0.33,
        beam_min_shape: 2.0, beam_max_shape: 4.0, beam_shape_power: 0.25,
        levels_autodim_temp: 0.5,
        // mask_amplify currently unused since pass3 outputs the dim product
        // (matching slang). Kept for backward compat / future bloom-fake port.
        mask_type: 0, mask_triad_size: 3.0, mask_amplify: 4.811320754717,
        mask_lut_enabled: 1,
        mask_sample_mode: 1,  // hardware-resample (validated path); see Status.md
        halation_weight: 0.0,
        diffusion_weight: 0.075,
        // Pass 4 defaults: slang-equivalent (flat mode, slang's border params,
        // no brightness compensation). Use SwiftRunner --geom-mode to flip on
        // curvature for visual experiments.
        geom_mode: 0,
        geom_radius_factor: 0.25,
        border_size: 0.015,
        border_darkness: 2.0,
        border_compress: 2.5,
        brightness_boost: 1.0,
        debug_pass_index: 0
    )
}

// MARK: - CLI

struct Args {
    var metalPath: String
    var inputPath: String
    var outDir: String
    var scale: Int        // Y-upscale factor: how much taller is output vs source.
    var neutralGamma: Bool // When true, crt_gamma == lcd_gamma => round-trip identity.
    var maskLutPath: String? // Large LUT (512x512, mipmapped) -- consumed by
                             // pass3 when mask_sample_mode=1.
    var maskLutSmallPath: String? // Small LUT (64x64, no mipmap) -- consumed
                             // by pass 5 (mask_resize_v) at slang-default
                             // settings. Required when mask_sample_mode=0.
    var maskType: UInt32     // 0 = aperture grille, 1 = slot mask, 2 = shadow mask.
                             // Sets uniforms.mask_type AND mask_amplify based on
                             // slang's mask_*_avg_color constants.
    var demoMode: Bool       // Flip on pass-4 curvature + brightness boost so
                             // the headless output looks like the RetroVisor
                             // app preview. Off by default = slang-equivalent
                             // for delta-E validation.
    var benchIterations: Int // When > 0, run the full pipeline this many times
                             // and print frame-time statistics (mean, p50,
                             // p95, min, max). PNG output is still written
                             // once, after the bench loop completes.
}

// mask_amplify for each mask type. Slang's get_mask_amplify() returns
// 1.0 / mask_*_avg_color (bind-shader-params.h:306-330, user-cgp-constants.h:40-46).
let maskAmplifyByType: [UInt32: Float] = [
    0: 255.0 / 53.0,   // grille15:    53/255 -> amplify ~ 4.811
    1: 255.0 / 46.0,   // slot:        46/255 -> amplify ~ 5.543
    2: 255.0 / 41.0,   // shadow EDP:  41/255 -> amplify ~ 6.220
]

func parseArgs() -> Args {
    var metal: String?, input: String?, out: String?, lut: String?, lutSmall: String?
    var scale = 4
    var neutral = false
    var demo = false
    var bench = 0
    var maskType: UInt32 = 0
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--metal":     metal = it.next()
        case "--input":     input = it.next()
        case "--outdir":    out = it.next()
        case "--scale":     scale = Int(it.next() ?? "4") ?? 4
        case "--neutral":   neutral = true
        case "--bench":     bench = Int(it.next() ?? "100") ?? 100
        case "--mask-lut":  lut = it.next()
        case "--mask-lut-small": lutSmall = it.next()
        case "--mask-type":
            switch it.next() ?? "" {
            case "grille", "0": maskType = 0
            case "slot",   "1": maskType = 1
            case "shadow", "2": maskType = 2
            default:
                FileHandle.standardError.write(Data("--mask-type must be grille|slot|shadow\n".utf8))
                exit(2)
            }
        case "--demo":      demo = true
        default:
            FileHandle.standardError.write(Data("unknown arg: \(a)\n".utf8))
            exit(2)
        }
    }
    guard let m = metal, let i = input, let o = out else {
        FileHandle.standardError.write(Data(
            "usage: SwiftRunner --metal X --input Y --outdir Z [--scale N] [--neutral] [--mask-lut PATH] [--mask-lut-small PATH] [--mask-type grille|slot|shadow] [--demo]\n".utf8))
        exit(2)
    }
    return Args(metalPath: m, inputPath: i, outDir: o,
                scale: scale, neutralGamma: neutral, maskLutPath: lut,
                maskLutSmallPath: lutSmall,
                maskType: maskType, demoMode: demo,
                benchIterations: bench)
}

// MARK: - Metal helpers

func makeDevice() -> MTLDevice {
    guard let dev = MTLCreateSystemDefaultDevice() else {
        FileHandle.standardError.write(Data("No Metal device available\n".utf8))
        exit(3)
    }
    return dev
}

func compileLibrary(device: MTLDevice, metalPath: String) -> MTLLibrary {
    let url = URL(fileURLWithPath: metalPath)
    guard let src = try? String(contentsOf: url, encoding: .utf8) else {
        FileHandle.standardError.write(Data("Failed to read metal source at \(metalPath)\n".utf8))
        exit(4)
    }
    let opts = MTLCompileOptions()
    do {
        return try device.makeLibrary(source: src, options: opts)
    } catch {
        FileHandle.standardError.write(Data("Metal compile failed: \(error)\n".utf8))
        exit(5)
    }
}

func makeKernel(library: MTLLibrary, device: MTLDevice, name: String) -> MTLComputePipelineState {
    guard let f = library.makeFunction(name: name) else {
        FileHandle.standardError.write(Data("Function \(name) not found in library\n".utf8))
        exit(6)
    }
    do {
        return try device.makeComputePipelineState(function: f)
    } catch {
        FileHandle.standardError.write(Data("Pipeline creation failed for \(name): \(error)\n".utf8))
        exit(7)
    }
}

func makeSampler(device: MTLDevice) -> MTLSamplerState {
    let d = MTLSamplerDescriptor()
    d.minFilter = .linear
    d.magFilter = .linear
    d.sAddressMode = .clampToEdge
    d.tAddressMode = .clampToEdge
    return device.makeSamplerState(descriptor: d)!
}

func makeRepeatSampler(device: MTLDevice) -> MTLSamplerState {
    let d = MTLSamplerDescriptor()
    d.minFilter = .linear
    d.magFilter = .linear
    d.mipFilter = .linear         // Match slang's mask_grille_texture_large_mipmap=true
    d.sAddressMode = .repeat
    d.tAddressMode = .repeat
    d.maxAnisotropy = 16          // Match librashader/RetroArch default anisotropy
    return device.makeSamplerState(descriptor: d)!
}

func makeTexture(device: MTLDevice, width: Int, height: Int,
                 pixelFormat: MTLPixelFormat,
                 usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> MTLTexture
{
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: pixelFormat, width: width, height: height, mipmapped: false)
    d.usage = usage
    d.storageMode = .shared
    return device.makeTexture(descriptor: d)!
}

// MARK: - PNG IO

func loadPNGAsTexture(device: MTLDevice, path: String,
                       mipmapped: Bool = false,
                       queue: MTLCommandQueue? = nil) -> MTLTexture {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        FileHandle.standardError.write(Data("Failed to load \(path)\n".utf8))
        exit(8)
    }
    let w = cg.width
    let h = cg.height
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let bytesPerRow = w * 4
    var bytes = [UInt8](repeating: 0, count: bytesPerRow * h)
    bytes.withUnsafeMutableBytes { ptr in
        let ctx = CGContext(
            data: ptr.baseAddress,
            width: w, height: h,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    }

    // sample() with .bgra8Unorm expects BGRA byte order; swap RGBA -> BGRA.
    for i in stride(from: 0, to: bytes.count, by: 4) {
        let r = bytes[i]; bytes[i] = bytes[i + 2]; bytes[i + 2] = r
    }

    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: mipmapped)
    d.usage = [.shaderRead]
    d.storageMode = .private  // .private allows blit-generated mipmaps; we'll
                              // stage via a temporary .shared texture below.
    if !mipmapped {
        d.storageMode = .shared
        let tex = device.makeTexture(descriptor: d)!
        bytes.withUnsafeBytes {
            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                        withBytes: $0.baseAddress!, bytesPerRow: bytesPerRow)
        }
        return tex
    }
    // Mipmapped path: stage mip 0 into a CPU-visible texture, blit to a
    // GPU-private mipmapped destination, then have the GPU build the chain.
    let staging = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    staging.usage = [.shaderRead]
    staging.storageMode = .shared
    let src = device.makeTexture(descriptor: staging)!
    bytes.withUnsafeBytes {
        src.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                    withBytes: $0.baseAddress!, bytesPerRow: bytesPerRow)
    }
    let dst = device.makeTexture(descriptor: d)!
    guard let q = queue else {
        FileHandle.standardError.write(Data(
            "loadPNGAsTexture(mipmapped:true) requires a command queue\n".utf8))
        exit(8)
    }
    let buf = q.makeCommandBuffer()!
    let blit = buf.makeBlitCommandEncoder()!
    blit.copy(from: src, sourceSlice: 0, sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1),
              to: dst, destinationSlice: 0, destinationLevel: 0,
              destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    blit.generateMipmaps(for: dst)
    blit.endEncoding()
    buf.commit()
    buf.waitUntilCompleted()
    return dst
}

func writeTextureAsPNG(_ texture: MTLTexture, path: String) {
    let w = texture.width
    let h = texture.height
    let bytesPerRow = w * 4
    var bytes = [UInt8](repeating: 0, count: bytesPerRow * h)
    bytes.withUnsafeMutableBytes { ptr in
        texture.getBytes(ptr.baseAddress!,
                         bytesPerRow: bytesPerRow,
                         from: MTLRegionMake2D(0, 0, w, h),
                         mipmapLevel: 0)
    }
    if texture.pixelFormat == .bgra8Unorm || texture.pixelFormat == .bgra8Unorm_srgb {
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let b = bytes[i]; bytes[i] = bytes[i + 2]; bytes[i + 2] = b
        }
    }
    let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    let provider = CGDataProvider(data: Data(bytes) as CFData)!
    let cg = CGImage(width: w, height: h,
                     bitsPerComponent: 8, bitsPerPixel: 32,
                     bytesPerRow: bytesPerRow,
                     space: cs, bitmapInfo: info,
                     provider: provider, decode: nil,
                     shouldInterpolate: false, intent: .defaultIntent)!
    let url = URL(fileURLWithPath: path)
    let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                               UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
}

// MARK: - Pipeline dispatch

func runKernel(commandBuffer: MTLCommandBuffer,
               pipeline: MTLComputePipelineState,
               sampler: MTLSamplerState,
               source: MTLTexture, target: MTLTexture,
               uniforms: inout Uniforms,
               extraTextures: [(MTLTexture, Int)] = [],
               extraSamplers: [(MTLSamplerState, Int)] = [])
{
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(source, index: 0)
    encoder.setTexture(target, index: 1)
    for (tex, idx) in extraTextures { encoder.setTexture(tex, index: idx) }
    encoder.setSamplerState(sampler, index: 0)
    for (sam, idx) in extraSamplers { encoder.setSamplerState(sam, index: idx) }
    encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
    let tg = MTLSize(width: 16, height: 16, depth: 1)
    let groups = MTLSize(
        width: (target.width + tg.width - 1) / tg.width,
        height: (target.height + tg.height - 1) / tg.height,
        depth: 1)
    encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    encoder.endEncoding()
}

// MARK: - main

let args = parseArgs()
try? FileManager.default.createDirectory(atPath: args.outDir,
                                         withIntermediateDirectories: true)

let device = makeDevice()
let library = compileLibrary(device: device, metalPath: args.metalPath)
let pass1     = makeKernel(library: library, device: device, name: "crt_royale::pass1_linearize")
let pass2     = makeKernel(library: library, device: device, name: "crt_royale::pass2_vertical_scanlines")
let pass3     = makeKernel(library: library, device: device, name: "crt_royale::pass3_apply_mask")
let finalEnc  = makeKernel(library: library, device: device, name: "crt_royale::pass_final_encode")
let pass4     = makeKernel(library: library, device: device, name: "crt_royale::pass4_geometry_aa")
let bloomApxK = makeKernel(library: library, device: device, name: "crt_royale::pass_bloom_approx")
let halVK     = makeKernel(library: library, device: device, name: "crt_royale::pass_halation_v")
let halHK     = makeKernel(library: library, device: device, name: "crt_royale::pass_halation_h")
let mrVK      = makeKernel(library: library, device: device, name: "crt_royale::pass_mask_resize_v")
let mrHK      = makeKernel(library: library, device: device, name: "crt_royale::pass_mask_resize_h")
let brightpsK = makeKernel(library: library, device: device, name: "crt_royale::pass_brightpass")
let bloomVK   = makeKernel(library: library, device: device, name: "crt_royale::pass_bloom_v")
let bloomHK   = makeKernel(library: library, device: device, name: "crt_royale::pass_bloom_h_reconstitute")
let linearExp = makeKernel(library: library, device: device, name: "crt_royale::pass_linear_export")
let sampler       = makeSampler(device: device)
let repeatSampler = makeRepeatSampler(device: device)
let queue         = device.makeCommandQueue()!

let input = loadPNGAsTexture(device: device, path: args.inputPath)
let w = input.width
let h = input.height
let outH = h * args.scale          // Y-upscale to mimic real CRT/LCD pixel pitch.

// Optional phosphor-mask LUT (mask_grille_texture_large from the slang
// shaders repo). When provided, pass3 samples it at a slang-equivalent tiled
// UV and ignores the procedural aperture grille. The PNG bytes are already
// linear ("TileableLinear*.png"); we load them into a non-sRGB texture so
// they reach the sampler verbatim. Mipmaps are essential -- the slang preset
// sets mask_grille_texture_large_mipmap=true, and at our triad sizes the
// sampler's LOD selection lives at mip levels 4-5, smoothing the subpixel
// structure (without mipmaps we'd get aliased sharp triads).
let maskLut: MTLTexture?
if let p = args.maskLutPath, FileManager.default.fileExists(atPath: p) {
    maskLut = loadPNGAsTexture(device: device, path: p,
                                mipmapped: true, queue: queue)
    print("[runner] mask_lut: \(p) (\(maskLut!.width)x\(maskLut!.height), mipmapped)")
} else {
    maskLut = nil
    print("[runner] mask_lut: <none> (procedural fallback)")
}

// Small LUT (64x64, no mipmap) for slang Pass 5 input.
let maskLutSmall: MTLTexture?
if let p = args.maskLutSmallPath, FileManager.default.fileExists(atPath: p) {
    maskLutSmall = loadPNGAsTexture(device: device, path: p,
                                     mipmapped: false, queue: queue)
    print("[runner] mask_lut_small: \(p) (\(maskLutSmall!.width)x\(maskLutSmall!.height))")
} else {
    maskLutSmall = nil
    print("[runner] mask_lut_small: <none> (mask-resize V/H will be skipped)")
}

print("[runner] input: \(args.inputPath) (\(w)x\(h))  scale=\(args.scale)x  neutral=\(args.neutralGamma)")
print("[runner] Uniforms.stride = \(MemoryLayout<Uniforms>.stride)")

// Pass 1 keeps source resolution (matches scale_type=source in the slang
// preset). Pass 2 writes into the Y-upscaled output FBO -- this is what
// produces visible scanlines (pixel_height = video.y / output.y < 1).
// Pass 3 stays at the same upscaled resolution as Pass 2.
let linearized        = makeTexture(device: device, width: w, height: h,    pixelFormat: .rgba16Float)
let scanlinesVertical = makeTexture(device: device, width: w, height: outH, pixelFormat: .rgba16Float)
let maskedScanlines   = makeTexture(device: device, width: w, height: outH, pixelFormat: .rgba16Float)
let displayPass1      = makeTexture(device: device, width: w, height: h,    pixelFormat: .bgra8Unorm)
let displayPass2      = makeTexture(device: device, width: w, height: outH, pixelFormat: .bgra8Unorm)
let displayPass3      = makeTexture(device: device, width: w, height: outH, pixelFormat: .bgra8Unorm)
// BLOOM_APPROX: slang pass 2, absolute 320x240 (per crt-royale.slangp).
let bloomApprox       = makeTexture(device: device, width: 320, height: 240, pixelFormat: .rgba16Float)
let displayBloomApprox = makeTexture(device: device, width: 320, height: 240, pixelFormat: .bgra8Unorm)
// HALATION_V (slang pass 3) + HALATION_BLUR (slang pass 4): 9-tap separable
// Gauss blurs of BLOOM_APPROX. scale_type=source, scale=1.0 -> 320x240.
let halationV          = makeTexture(device: device, width: 320, height: 240, pixelFormat: .rgba16Float)
let halationBlur       = makeTexture(device: device, width: 320, height: 240, pixelFormat: .rgba16Float)
let displayHalationV   = makeTexture(device: device, width: 320, height: 240, pixelFormat: .bgra8Unorm)
let displayHalationBlur = makeTexture(device: device, width: 320, height: 240, pixelFormat: .bgra8Unorm)
// MASK_RESIZE_V (slang pass 5) + MASK_RESIZE (slang pass 6).
// Pass 5: scale_x=64 abs, scale_y=viewport*0.0625 = outH*0.0625.
// Pass 6: scale_x=viewport*0.0625 = w*0.0625, scale_y=source = pass5_y.
let mrY     = max(Int((Float(outH) * 0.0625).rounded()), 2)
let mrXh    = max(Int((Float(w)    * 0.0625).rounded()), 2)
let maskResizeV   = makeTexture(device: device, width: 64,   height: mrY, pixelFormat: .rgba16Float)
let maskResize    = makeTexture(device: device, width: mrXh, height: mrY, pixelFormat: .rgba16Float)
let displayMaskRV = makeTexture(device: device, width: 64,   height: mrY, pixelFormat: .bgra8Unorm)
let displayMaskR  = makeTexture(device: device, width: mrXh, height: mrY, pixelFormat: .bgra8Unorm)
// BRIGHTPASS: slang pass 8, viewport-scale (= MASKED_SCANLINES size).
// Sampled later by bloom V/H to produce BLOOM_FINAL.
let brightpass         = makeTexture(device: device, width: w, height: outH, pixelFormat: .rgba16Float)
let displayBrightpass  = makeTexture(device: device, width: w, height: outH, pixelFormat: .bgra8Unorm)
// BLOOM_V (slang pass 9): vertical Gaussian blur of brightpass.
let bloomV             = makeTexture(device: device, width: w, height: outH, pixelFormat: .rgba16Float)
let displayBloomV      = makeTexture(device: device, width: w, height: outH, pixelFormat: .bgra8Unorm)
// BLOOM_FINAL (slang pass 10): horizontal blur + reconstitute. Final input
// to slang pass 11; Pass 4 now samples this instead of MASKED_SCANLINES.
let bloomFinal         = makeTexture(device: device, width: w, height: outH, pixelFormat: .rgba16Float)
let displayBloomFinal  = makeTexture(device: device, width: w, height: outH, pixelFormat: .bgra8Unorm)

var uniforms = Uniforms.defaults
if args.neutralGamma {
    // Disable the CRT-vs-LCD gamma mismatch so pass1 round-trips to identity.
    uniforms.crt_gamma = uniforms.lcd_gamma
}
uniforms.texture_size = SIMD2<Float>(Float(w), Float(h))
uniforms.video_size   = SIMD2<Float>(Float(w), Float(h))
uniforms.output_size  = SIMD2<Float>(Float(w), Float(outH))
uniforms.mask_lut_enabled = (maskLut != nil) ? 1 : 0
uniforms.mask_type    = args.maskType
uniforms.mask_amplify = maskAmplifyByType[args.maskType] ?? uniforms.mask_amplify
if args.demoMode {
    uniforms.geom_mode         = 1
    uniforms.geom_radius_factor = 0.20
    uniforms.brightness_boost  = 5.0
}

// Core pipeline as a closure so we can run it multiple times for benchmarks.
// Each call records all 12 CRT-Royale passes + pass4_geometry_aa into a fresh
// command buffer, commits + waits, returns GPU-time in milliseconds. The
// per-pass linearExp dumps used for snapshot export are excluded here -- bench
// measures the actual user-facing pipeline cost only.
func runCorePipeline() -> Double {
    let cmd = queue.makeCommandBuffer()!
    // Pass 1 -> linearized
    runKernel(commandBuffer: cmd, pipeline: pass1, sampler: sampler,
              source: input, target: linearized, uniforms: &uniforms)
    // Pass 2 (BLOOM_APPROX)
    runKernel(commandBuffer: cmd, pipeline: bloomApxK, sampler: sampler,
              source: linearized, target: bloomApprox, uniforms: &uniforms)
    // Pass 3 (HALATION_V)
    runKernel(commandBuffer: cmd, pipeline: halVK, sampler: sampler,
              source: bloomApprox, target: halationV, uniforms: &uniforms)
    // Pass 4 (HALATION_BLUR)
    runKernel(commandBuffer: cmd, pipeline: halHK, sampler: sampler,
              source: halationV, target: halationBlur, uniforms: &uniforms)
    // Pass 5 + 6 (MASK_RESIZE V/H) -- only if small LUT bound
    if let lutSmall = maskLutSmall {
        runKernel(commandBuffer: cmd, pipeline: mrVK, sampler: repeatSampler,
                  source: lutSmall, target: maskResizeV, uniforms: &uniforms)
        runKernel(commandBuffer: cmd, pipeline: mrHK, sampler: repeatSampler,
                  source: maskResizeV, target: maskResize, uniforms: &uniforms)
    }
    // Pass 1 (vertical scanlines, slang Pass 1)
    runKernel(commandBuffer: cmd, pipeline: pass2, sampler: sampler,
              source: linearized, target: scanlinesVertical, uniforms: &uniforms)
    // Pass 7 (apply-mask)
    var p3Extras: [(MTLTexture, Int)] = [(halationBlur, 3), (maskResize, 4)]
    var p3Samplers: [(MTLSamplerState, Int)] = []
    if let lut = maskLut {
        p3Extras.append((lut, 2))
        p3Samplers.append((repeatSampler, 1))
    }
    runKernel(commandBuffer: cmd, pipeline: pass3, sampler: sampler,
              source: scanlinesVertical, target: maskedScanlines, uniforms: &uniforms,
              extraTextures: p3Extras, extraSamplers: p3Samplers)
    // Pass 8 (BRIGHTPASS)
    runKernel(commandBuffer: cmd, pipeline: brightpsK, sampler: sampler,
              source: maskedScanlines, target: brightpass, uniforms: &uniforms,
              extraTextures: [(bloomApprox, 2)])
    // Pass 9 (BLOOM_V)
    runKernel(commandBuffer: cmd, pipeline: bloomVK, sampler: sampler,
              source: brightpass, target: bloomV, uniforms: &uniforms)
    // Pass 10 (BLOOM_FINAL)
    runKernel(commandBuffer: cmd, pipeline: bloomHK, sampler: sampler,
              source: bloomV, target: bloomFinal, uniforms: &uniforms,
              extraTextures: [(brightpass, 2), (maskedScanlines, 3),
                              (halationBlur, 4)])
    // Pass 11 (Pass 4 here: geometry + AA + final encode)
    let displayFinalScratch = makeTexture(device: device, width: w, height: outH,
                                          pixelFormat: .bgra8Unorm)
    runKernel(commandBuffer: cmd, pipeline: pass4, sampler: sampler,
              source: bloomFinal, target: displayFinalScratch, uniforms: &uniforms)
    cmd.commit()
    cmd.waitUntilCompleted()
    // gpuEndTime/gpuStartTime are CFAbsoluteTime in seconds.
    return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0
}

// Benchmark mode: run N times, print stats, then proceed to snapshot output.
if args.benchIterations > 0 {
    let N = args.benchIterations
    // Warmup -- excluded from stats. Compiles kernels, primes caches.
    for _ in 0..<3 { _ = runCorePipeline() }
    var samples: [Double] = []
    samples.reserveCapacity(N)
    for _ in 0..<N { samples.append(runCorePipeline()) }
    samples.sort()
    let mean = samples.reduce(0, +) / Double(N)
    let p50 = samples[N/2]
    let p95 = samples[min(N - 1, Int(Double(N) * 0.95))]
    let mn = samples.first!
    let mx = samples.last!
    print("[bench] frames=\(N) at \(w)x\(outH) -- " +
          "mean=\(String(format: "%.3f", mean)) ms, " +
          "p50=\(String(format: "%.3f", p50)) ms, " +
          "p95=\(String(format: "%.3f", p95)) ms, " +
          "min=\(String(format: "%.3f", mn)) ms, " +
          "max=\(String(format: "%.3f", mx)) ms, " +
          "fps_p50=\(String(format: "%.1f", 1000.0 / p50))")
}

let cmd = queue.makeCommandBuffer()!

// Pass 1 -> linearized
runKernel(commandBuffer: cmd, pipeline: pass1, sampler: sampler,
          source: input, target: linearized, uniforms: &uniforms)

// BLOOM_APPROX (slang pass 2) -> bloomApprox 320x240.
runKernel(commandBuffer: cmd, pipeline: bloomApxK, sampler: sampler,
          source: linearized, target: bloomApprox, uniforms: &uniforms)

// HALATION_V (slang pass 3) -> halationV. Vertical 9-tap Gauss of bloomApprox.
runKernel(commandBuffer: cmd, pipeline: halVK, sampler: sampler,
          source: bloomApprox, target: halationV, uniforms: &uniforms)

// HALATION_BLUR (slang pass 4) -> halationBlur. Horizontal 9-tap Gauss of halationV.
runKernel(commandBuffer: cmd, pipeline: halHK, sampler: sampler,
          source: halationV, target: halationBlur, uniforms: &uniforms)

// MASK_RESIZE_V (slang pass 5) + MASK_RESIZE (slang pass 6).
// Lanczos-sinc downsample of the small 64x64 LUT. Pass 5 uses
// repeatSampler so the per-tile UV wraps cleanly across the LUT.
if let lutSmall = maskLutSmall {
    runKernel(commandBuffer: cmd, pipeline: mrVK, sampler: repeatSampler,
              source: lutSmall, target: maskResizeV, uniforms: &uniforms)
    runKernel(commandBuffer: cmd, pipeline: mrHK, sampler: repeatSampler,
              source: maskResizeV, target: maskResize, uniforms: &uniforms)
}

// Pass 2 -> scanlinesVertical
runKernel(commandBuffer: cmd, pipeline: pass2, sampler: sampler,
          source: linearized, target: scanlinesVertical, uniforms: &uniforms)

// Pass 3 -> maskedScanlines. Texture bindings:
//   tex 0 = scanlinesVertical (source)
//   tex 1 = maskedScanlines   (target)
//   tex 2 = large LUT         (optional, for mask_sample_mode=1)
//   tex 3 = HALATION_BLUR     (always)
//   tex 4 = MASK_RESIZE       (always when available, for mode=0)
// Sampler 0 = linear/clamp; sampler 1 = linear/repeat/mipmap for LUT.
var pass3Extras: [(MTLTexture, Int)] = [(halationBlur, 3),
                                         (maskResize,   4)]
var pass3Samplers: [(MTLSamplerState, Int)] = []
if let lut = maskLut {
    pass3Extras.append((lut, 2))
    pass3Samplers.append((repeatSampler, 1))
}
runKernel(commandBuffer: cmd, pipeline: pass3, sampler: sampler,
          source: scanlinesVertical, target: maskedScanlines, uniforms: &uniforms,
          extraTextures: pass3Extras, extraSamplers: pass3Samplers)

// BRIGHTPASS (slang pass 8): area-based brightness extraction. Reads
// MASKED_SCANLINES (texture 0) and BLOOM_APPROX (texture 2). Output is the
// "bloom source" -- the fraction of MASKED_SCANLINES that should bloom into
// neighbors.
runKernel(commandBuffer: cmd, pipeline: brightpsK, sampler: sampler,
          source: maskedScanlines, target: brightpass, uniforms: &uniforms,
          extraTextures: [(bloomApprox, 2)])

// BLOOM_V (slang pass 9): vertical 9-tap Gaussian blur of BRIGHTPASS.
runKernel(commandBuffer: cmd, pipeline: bloomVK, sampler: sampler,
          source: brightpass, target: bloomV, uniforms: &uniforms)

// BLOOM_H_RECONSTITUTE (slang pass 10): horizontal blur of bloomV + add
// dimpass back. Output is BLOOM_FINAL, the input to slang pass 11.
// Inputs: bloomV (tex 0), brightpass (tex 2), masked_scanlines (tex 3),
// halation_blur (tex 4) -- the last one feeds the diffusion_weight mix.
runKernel(commandBuffer: cmd, pipeline: bloomHK, sampler: sampler,
          source: bloomV, target: bloomFinal, uniforms: &uniforms,
          extraTextures: [(brightpass, 2), (maskedScanlines, 3),
                          (halationBlur, 4)])

// Per-pass snapshots are written as raw linear values (no gamma encoding)
// to match librashader's per-pass output format -- it exports the linear
// floats stored in sRGB framebuffers as PNG bytes, not the sRGB-encoded
// bytes. Apples-to-apples comparison requires the same convention.
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: linearized, target: displayPass1, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: scanlinesVertical, target: displayPass2, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: maskedScanlines, target: displayPass3, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: bloomApprox, target: displayBloomApprox, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: halationV, target: displayHalationV, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: halationBlur, target: displayHalationBlur, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: maskResizeV, target: displayMaskRV, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: maskResize, target: displayMaskR, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: brightpass, target: displayBrightpass, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: bloomV, target: displayBloomV, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: linearExp, sampler: sampler,
          source: bloomFinal, target: displayBloomFinal, uniforms: &uniforms)
// "Final" goes through pass4_geometry_aa: border-dim, optional barrel
// distortion (geom_mode=1), display gamma. With BLOOM_FINAL now available,
// Pass 4 samples it (= slang's pass 11 input) instead of MASKED_SCANLINES.
let displayFinal = makeTexture(device: device, width: w, height: outH,
                               pixelFormat: .bgra8Unorm)
runKernel(commandBuffer: cmd, pipeline: pass4, sampler: sampler,
          source: bloomFinal, target: displayFinal, uniforms: &uniforms)

cmd.commit()
cmd.waitUntilCompleted()

if let err = cmd.error {
    FileHandle.standardError.write(Data("Command buffer error: \(err)\n".utf8))
    exit(9)
}

let inputCopy = "\(args.outDir)/00-input.png"
let p1Out     = "\(args.outDir)/01-pass1.png"
let p2Out     = "\(args.outDir)/02-pass2.png"
let p2bOut    = "\(args.outDir)/02b-bloom_approx.png"
let p2cOut    = "\(args.outDir)/02c-halation_v.png"
let p2dOut    = "\(args.outDir)/02d-halation_blur.png"
let p2eOut    = "\(args.outDir)/02e-mask_resize_v.png"
let p2fOut    = "\(args.outDir)/02f-mask_resize.png"
let p3Out     = "\(args.outDir)/03-pass3.png"
let p3bOut    = "\(args.outDir)/03b-brightpass.png"
let p3cOut    = "\(args.outDir)/03c-bloom_v.png"
let p3dOut    = "\(args.outDir)/03d-bloom_final.png"
let finalOut  = "\(args.outDir)/04-final.png"

writeTextureAsPNG(input,               path: inputCopy)
writeTextureAsPNG(displayPass1,        path: p1Out)
writeTextureAsPNG(displayPass2,        path: p2Out)
writeTextureAsPNG(displayBloomApprox,  path: p2bOut)
writeTextureAsPNG(displayHalationV,    path: p2cOut)
writeTextureAsPNG(displayHalationBlur, path: p2dOut)
writeTextureAsPNG(displayMaskRV,       path: p2eOut)
writeTextureAsPNG(displayMaskR,        path: p2fOut)
writeTextureAsPNG(displayPass3,        path: p3Out)
writeTextureAsPNG(displayBrightpass,   path: p3bOut)
writeTextureAsPNG(displayBloomV,       path: p3cOut)
writeTextureAsPNG(displayBloomFinal,   path: p3dOut)
writeTextureAsPNG(displayFinal,        path: finalOut)

print("[runner] wrote: \(inputCopy)")
print("[runner] wrote: \(p1Out)")
print("[runner] wrote: \(p2Out)")
print("[runner] wrote: \(p2bOut)")
print("[runner] wrote: \(p2cOut)")
print("[runner] wrote: \(p2dOut)")
print("[runner] wrote: \(p2eOut)")
print("[runner] wrote: \(p2fOut)")
print("[runner] wrote: \(p3Out)")
print("[runner] wrote: \(p3bOut)")
print("[runner] wrote: \(p3cOut)")
print("[runner] wrote: \(p3dOut)")
print("[runner] wrote: \(finalOut)")
print("[runner] OK")
