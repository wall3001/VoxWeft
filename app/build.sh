#!/bin/bash
# Build VoxWeftApp without Xcode (Command Line Tools only).
# SwiftPM's manifest linking is frequently broken on CLT-only installs;
# this compiles the sources directly and lays out a minimal .app bundle.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-build}
APP="$OUT/VoxWeft.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/intro"
echo "compiling…"
swiftc -O -parse-as-library -target arm64-apple-macosx14.0 \
  -framework SwiftUI -framework AVFoundation -framework Accelerate \
  Sources/VoxWeftApp/*.swift -o "$APP/Contents/MacOS/VoxWeftApp"
cp Sources/VoxWeftApp/intro/*.txt "$APP/Contents/Resources/intro/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>VoxWeft</string>
  <key>CFBundleIdentifier</key><string>ai.openbmb.voxweft</string>
  <key>CFBundleExecutable</key><string>VoxWeftApp</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSMicrophoneUsageDescription</key><string>VoxWeft listens to translate your speech on this machine.</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
echo "built $APP"
echo "run:   open $APP        (or $APP/Contents/MacOS/VoxWeftApp)"
