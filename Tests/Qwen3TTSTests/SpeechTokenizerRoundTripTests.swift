import XCTest
import MLX
import AudioCommon
@testable import Qwen3TTS

/// The real tokenizer on a real 12 s clip. Skipped when the tokenizer weights are not in the cache.
final class SpeechTokenizerRoundTripTests: XCTestCase {
    static let tokenizerDirectory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Caches/qwen3-speech/models/Qwen/Qwen3-TTS-Tokenizer-12Hz")

    nonisolated(unsafe) static var loaded: (SpeechTokenizerEncoder, SpeechTokenizerDecoder, MLXArray)?

    func tokenizer() throws -> (encoder: SpeechTokenizerEncoder, decoder: SpeechTokenizerDecoder, codes: MLXArray) {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.tokenizerDirectory.appendingPathComponent("model.safetensors").path),
            "tokenizer weights are not in \(Self.tokenizerDirectory.path)")
        if let loaded = Self.loaded { return loaded }
        let config = SpeechTokenizerDecoderConfig()
        let encoder = SpeechTokenizerEncoder(config: config)
        try TTSWeightLoader.loadSpeechTokenizerEncoderWeights(into: encoder, from: Self.tokenizerDirectory)
        let decoder = SpeechTokenizerDecoder(config: config)
        try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(into: decoder, from: Self.tokenizerDirectory)
        let codes = encoder.encode(samples: try Self.clip())
        eval(codes)
        Self.loaded = (encoder, decoder, codes)
        return (encoder, decoder, codes)
    }

    static func clip() throws -> [Float] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "tovarisch", withExtension: "wav"))
        let (samples, rate) = try AudioFileLoader.loadWAV(url: url)
        XCTAssertEqual(rate, 24_000)
        return samples
    }

    func testTheClipIsOneHundredFiftyFramesAndDecodesToItsOwnLength() throws {
        let (_, decoder, codes) = try tokenizer()
        XCTAssertEqual(try Self.clip().count, 288_000)
        XCTAssertEqual(codes.dim(2), 150)
        XCTAssertEqual(decoder.decode(codes: codes).count, 288_000)
    }

    func testSmallChunksMatchTheOnePassDecodeInLengthWithNoJumpAtTheSeams() throws {
        let (_, decoder, codes) = try tokenizer()
        let onePass = decoder.decode(codes: codes)
        let chunkFrames = 25
        let chunked = decoder.chunkedDecode(codes: codes, chunkSize: chunkFrames, leftContext: 25).squeezed()
        eval(chunked)
        let samples = chunked.asArray(Float.self)
        XCTAssertEqual(samples.count, onePass.count)

        let onePassMaxStep = zip(onePass.dropFirst(), onePass).map { abs($0 - $1) }.max() ?? 0
        let seams = stride(from: chunkFrames * decoder.samplesPerFrame, to: samples.count, by: chunkFrames * decoder.samplesPerFrame)
        for seam in seams {
            XCTAssertLessThanOrEqual(abs(samples[seam] - samples[seam - 1]), onePassMaxStep, "seam at sample \(seam)")
        }
    }
}
