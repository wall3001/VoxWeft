# VoxWeft macOS shell

A thin SwiftUI client for `voxweft serve`. Hold-to-talk, streamed 48 kHz
playback with a 200 ms jitter buffer, per-turn latency chips (ASR / MT /
first audio / total, measured client-side), and a mic-free demo that feeds the
bundled repo intro through the same pipeline — handy for screen recordings.

~500 lines, three files. It is disposable by design: the protocol lives in
`voxweft/runtime/server.py`, not here.

## Build

With Xcode:

```bash
cd app && rm -rf .build && swift build -c release
.build/release/VoxWeftApp
```

With Command Line Tools only (SwiftPM manifest linking is often broken on
CLT installs; this compiles with `swiftc` and lays out a minimal `.app`):

```bash
cd app && ./build.sh && open build/VoxWeft.app
```

`rm -rf .build` before rebuilding after copying sources onto another machine:
SwiftPM trusts mtimes and will report "Build complete" in ~3 s without
compiling anything. A real build is 30+ s.

## Run

1. `voxweft bridge` and `voxweft serve` running (see the root README).
2. Launch the app, set the URL (`ws://127.0.0.1:3100/ws/voice/v2`), Connect.
   macOS will ask for microphone permission once.
3. Pick mode and languages. Hold the big button to talk; release to translate.
   `▶︎ Play repo intro` runs the six bundled sentences without a microphone.

## Recording a demo

```bash
screencapture -v -V 60 demo.mov        # 60 s of the whole screen, with system audio
```

or QuickTime › File › New Screen Recording. Playback goes through the default
output device, so it is captured. Keep the latency chips in frame — they are
the point.

## Notes

* No SwiftUI `Button` in this app. On macOS 26 with Swift 6.3 a plain Button
  click aborts inside `_ButtonGesture` (DEVLOG §7.2); everything is a gesture.
* Capture and playback share one `AVAudioEngine` with voice processing on, so
  the system AEC has the playback signal as reference. Two engines = no AEC.
