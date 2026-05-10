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
    var debug_pass_index: UInt32

    static let defaults = Uniforms(
        crt_gamma: 2.5, lcd_gamma: 2.2,
        interlace_detect: 1, interlace_bff: 0, interlace_1080i: 0,
        frame_count: 1,
        texture_size: [0, 0], video_size: [0, 0], output_size: [0, 0],
        beam_min_sigma: 0.02, beam_max_sigma: 0.30, beam_spot_power: 0.33,
        beam_min_shape: 2.0, beam_max_shape: 4.0, beam_shape_power: 0.25,
        levels_autodim_temp: 0.5,
        mask_type: 0, mask_triad_size: 3.0, mask_amplify: 3.0,
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
}

func parseArgs() -> Args {
    var metal: String?, input: String?, out: String?
    var scale = 4
    var neutral = false
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--metal":   metal = it.next()
        case "--input":   input = it.next()
        case "--outdir":  out = it.next()
        case "--scale":   scale = Int(it.next() ?? "4") ?? 4
        case "--neutral": neutral = true
        default:
            FileHandle.standardError.write(Data("unknown arg: \(a)\n".utf8))
            exit(2)
        }
    }
    guard let m = metal, let i = input, let o = out else {
        FileHandle.standardError.write(Data(
            "usage: SwiftRunner --metal X --input Y --outdir Z [--scale N] [--neutral]\n".utf8))
        exit(2)
    }
    return Args(metalPath: m, inputPath: i, outDir: o,
                scale: scale, neutralGamma: neutral)
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

func loadPNGAsTexture(device: MTLDevice, path: String) -> MTLTexture {
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

    let tex = makeTexture(device: device, width: w, height: h,
                          pixelFormat: .bgra8Unorm,
                          usage: [.shaderRead])
    bytes.withUnsafeBytes {
        tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                    withBytes: $0.baseAddress!, bytesPerRow: bytesPerRow)
    }
    return tex
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
               uniforms: inout Uniforms)
{
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(source, index: 0)
    encoder.setTexture(target, index: 1)
    encoder.setSamplerState(sampler, index: 0)
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
let sampler   = makeSampler(device: device)
let queue     = device.makeCommandQueue()!

let input = loadPNGAsTexture(device: device, path: args.inputPath)
let w = input.width
let h = input.height
let outH = h * args.scale          // Y-upscale to mimic real CRT/LCD pixel pitch.

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

var uniforms = Uniforms.defaults
if args.neutralGamma {
    // Disable the CRT-vs-LCD gamma mismatch so pass1 round-trips to identity.
    uniforms.crt_gamma = uniforms.lcd_gamma
}
uniforms.texture_size = SIMD2<Float>(Float(w), Float(h))
uniforms.video_size   = SIMD2<Float>(Float(w), Float(h))
uniforms.output_size  = SIMD2<Float>(Float(w), Float(outH))

let cmd = queue.makeCommandBuffer()!

// Pass 1 -> linearized
runKernel(commandBuffer: cmd, pipeline: pass1, sampler: sampler,
          source: input, target: linearized, uniforms: &uniforms)

// Pass 2 -> scanlinesVertical
runKernel(commandBuffer: cmd, pipeline: pass2, sampler: sampler,
          source: linearized, target: scanlinesVertical, uniforms: &uniforms)

// Pass 3 -> maskedScanlines (procedural aperture grille)
runKernel(commandBuffer: cmd, pipeline: pass3, sampler: sampler,
          source: scanlinesVertical, target: maskedScanlines, uniforms: &uniforms)

// Final encode of each linear-light stage so we can view it
runKernel(commandBuffer: cmd, pipeline: finalEnc, sampler: sampler,
          source: linearized, target: displayPass1, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: finalEnc, sampler: sampler,
          source: scanlinesVertical, target: displayPass2, uniforms: &uniforms)
runKernel(commandBuffer: cmd, pipeline: finalEnc, sampler: sampler,
          source: maskedScanlines, target: displayPass3, uniforms: &uniforms)

cmd.commit()
cmd.waitUntilCompleted()

if let err = cmd.error {
    FileHandle.standardError.write(Data("Command buffer error: \(err)\n".utf8))
    exit(9)
}

let inputCopy = "\(args.outDir)/00-input.png"
let p1Out     = "\(args.outDir)/01-pass1.png"
let p2Out     = "\(args.outDir)/02-pass2.png"
let p3Out     = "\(args.outDir)/03-pass3.png"
let finalOut  = "\(args.outDir)/04-final.png"

writeTextureAsPNG(input,         path: inputCopy)
writeTextureAsPNG(displayPass1,  path: p1Out)
writeTextureAsPNG(displayPass2,  path: p2Out)
writeTextureAsPNG(displayPass3,  path: p3Out)
writeTextureAsPNG(displayPass3,  path: finalOut) // final == pass3 in current pipeline

print("[runner] wrote: \(inputCopy)")
print("[runner] wrote: \(p1Out)")
print("[runner] wrote: \(p2Out)")
print("[runner] wrote: \(p3Out)")
print("[runner] wrote: \(finalOut)")
print("[runner] OK")
