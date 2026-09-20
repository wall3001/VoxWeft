// swift-tools-version: 5.9
import PackageDescription

// VoxWeft — thin macOS shell over `voxweft serve`.
// L3 wrapper: hold-to-talk, streamed playback, latency readout, mic-free demo.
// Build:  swift build -c release   (rm -rf .build first after copying sources —
// SwiftPM trusts mtimes and will report "Build complete" without compiling).
let package = Package(
    name: "VoxWeftApp",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "VoxWeftApp",
            path: "Sources/VoxWeftApp",
            resources: [.copy("intro"), .copy("../../Resources/logo.png"), .copy("../../Resources/logo@2x.png")]
        )
    ]
)
