// swift-tools-version: 6.0
import PackageDescription

// Stripped for dicta (https://github.com/jaxer/dicta, private): keeps only the Qwen3ASR and Qwen3TTS
// products and the two targets they need (AudioCommon, MLXCommon). Everything else upstream soniqo/speech-swift
// ships — every TTS/other-ASR/benchmark/server/CLI/MCP target, SpeechCore, SpeechVAD, and their
// dependencies (hummingbird, swift-nio, swift-sdk, WhisperKit, mlx-swift-lm, swift-system,
// swift-argument-parser) — has been removed. See README.md for upstream provenance.
//
// swift-tools-version bumped 5.10 -> 6.0 so every target defaults to Swift 6 language mode. No
// target opts back into `.swiftLanguageMode(.v5)` — this is deliberate, mirroring dicta's own
// build, and the `swiftSettings` below are explicit about it rather than relying on the implicit
// default.
let package = Package(
    name: "Qwen3Speech",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0")
    ],
    products: [
        .library(
            name: "Qwen3ASR",
            targets: ["Qwen3ASR"]
        ),
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6")
    ],
    targets: [
        .target(
            name: "AudioCommon",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers")
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "MLXCommon",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "Qwen3TTS",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
