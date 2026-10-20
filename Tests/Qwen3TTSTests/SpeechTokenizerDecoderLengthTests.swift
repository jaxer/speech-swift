import XCTest
import MLX
@testable import Qwen3TTS

/// The decoder's length rule, checked on untrained parameters: a decode of N codec frames is
/// exactly N × 1920 samples (`decode_upsample_rate` in the tokenizer's config.json).
final class SpeechTokenizerDecoderLengthTests: XCTestCase {
    static let decodeUpsampleRate = 1920

    func testSamplesPerFrameIsTheProductOfEveryUpsamplingStage() {
        XCTAssertEqual(SpeechTokenizerDecoderConfig().samplesPerFrame, Self.decodeUpsampleRate)
        XCTAssertEqual(SpeechTokenizerDecoder(config: .init()).samplesPerFrame, Self.decodeUpsampleRate)
    }

    /// The checkpoint's pre-upsample convolutions are `ConvTranspose1d(kernel=factor, stride=factor)`
    /// (`decoder.upsample.{0,1}.0.conv.weight` is [1024, 1024, 2]). A module built with a wider kernel
    /// trims more than the loaded weights produce, and every decode comes out short.
    func testPreUpsampleConvolutionsHaveTheCheckpointsKernel() {
        let decoder = SpeechTokenizerDecoder(config: .init())
        for (conv, ratio) in zip([decoder.preUpsample1, decoder.preUpsample2], decoder.config.upsamplingRatios) {
            XCTAssertEqual(conv.conv.weight.dim(1), ratio, "kernel size")
            XCTAssertEqual(conv.trimRight, 0)
        }
    }

    func testEveryDecodeIsExactlyFramesTimesSamplesPerFrame() {
        let decoder = SpeechTokenizerDecoder(config: .init())
        for frames in [1, 10, 25, 35, 36, 150, 193, 301, 400] {
            let codes = Self.randomCodes(frames: frames, config: decoder.config)
            let waveform = decoder.executeDecoder(codes)
            eval(waveform)
            XCTAssertEqual(waveform.dim(1), frames * Self.decodeUpsampleRate, "executeDecoder, \(frames) frames")
            XCTAssertEqual(decoder.decode(codes: codes).count, frames * Self.decodeUpsampleRate, "decode, \(frames) frames")
        }
    }

    func testChunkedDecodeWithSmallChunksKeepsTheLength() {
        let decoder = SpeechTokenizerDecoder(config: .init())
        let codes = Self.randomCodes(frames: 193, config: decoder.config)
        let waveform = decoder.chunkedDecode(codes: codes, chunkSize: 25, leftContext: 25)
        eval(waveform)
        XCTAssertEqual(waveform.dim(1), 193 * Self.decodeUpsampleRate)
    }

    /// Every pre-transformer layer is sliding-window attention over 72 frames
    /// (`layer_types` in the official config): through one layer, a frame never sees one more than
    /// 71 frames back.
    func testPreTransformerAttendsOnlyWithinTheSlidingWindow() {
        var config = SpeechTokenizerDecoderConfig()
        config.numLayers = 1
        let transformer = DecoderTransformer(config: config)
        let frames = 120
        let x = MLXRandom.normal([1, frames, config.latentDim])
        let changedFirstFrame = concatenated(
            [MLXRandom.normal([1, 1, config.latentDim]), x[0..., 1..., 0...]], axis: 1)
        let a = transformer(x)
        let b = transformer(changedFirstFrame)
        let window = config.slidingWindow
        let insideWindow = abs(a[0..., 0..<window, 0...] - b[0..., 0..<window, 0...]).max().item(Float.self)
        let beyondWindow = abs(a[0..., window..., 0...] - b[0..., window..., 0...]).max().item(Float.self)
        XCTAssertGreaterThan(insideWindow, 0)
        XCTAssertEqual(beyondWindow, 0)
    }

    static func randomCodes(frames: Int, config: SpeechTokenizerDecoderConfig) -> MLXArray {
        let upper = min(config.semanticCodebookSize, config.acousticCodebookSize)
        return MLXRandom.randInt(low: 0, high: upper, [1, config.numQuantizers, frames]).asType(.int32)
    }
}
