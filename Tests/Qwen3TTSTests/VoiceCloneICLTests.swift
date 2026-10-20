import XCTest
import MLX
import AudioCommon
@testable import Qwen3TTS

/// ICL voice cloning end to end with the 1.7B model. Skipped when its weights are not in the cache.
final class VoiceCloneICLTests: XCTestCase {
    static let modelId = "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit"
    static let referenceText = "У него каверзное. Ничего он, кроме мясного есть не может. А чтобы достать мясную пищу, он не может иначе поступать, как живое существо жизни лишить."

    func testTheClonedPhraseIsWholeCodecFrames() async throws {
        let modelDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/qwen3-speech/models/\(Self.modelId)")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent("config.json").path),
            "1.7B weights are not in \(modelDirectory.path)")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: SpeechTokenizerRoundTripTests.tokenizerDirectory.path),
            "tokenizer weights are not in the cache")

        let (model, encoder) = try await Qwen3TTSModel.fromPretrainedWithEncoder(
            modelId: Self.modelId, offlineMode: true)
        let reference = try SpeechTokenizerRoundTripTests.clip()
        let samples = model.synthesizeWithVoiceCloneICL(
            text: "Сегодня я проверил все тесты, и теперь голос звучит ровно, без пропусков и щелчков.",
            referenceAudio: reference, referenceSampleRate: 24_000,
            referenceText: Self.referenceText, language: "russian", codecEncoder: encoder)

        let samplesPerFrame = model.codecDecoder.samplesPerFrame
        XCTAssertGreaterThan(samples.count, 2 * 24_000)
        XCTAssertEqual(samples.count % samplesPerFrame, 0, "\(samples.count) samples")
    }
}
