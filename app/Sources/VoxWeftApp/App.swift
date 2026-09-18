import SwiftUI

@main
struct VoxWeftApp: App {
    @StateObject private var session = Session()

    var body: some Scene {
        WindowGroup("VoxWeft") {
            ContentView().environmentObject(session)
                .frame(minWidth: 820, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

// No SwiftUI `Button` anywhere in this app: on macOS 26 + Swift 6.3 a plain
// Button click aborts inside _ButtonGesture (DEVLOG §7.2). Taps and holds go
// through gestures instead.
private struct Tappable: ViewModifier {
    let action: () -> Void
    func body(content: Content) -> some View {
        content.contentShape(Rectangle()).onTapGesture(perform: action)
    }
}

extension View {
    func tappable(_ action: @escaping () -> Void) -> some View { modifier(Tappable(action: action)) }
}

struct Chip: View {
    let text: String
    var active = false
    var body: some View {
        Text(text)
            .font(.system(size: 13, weight: .medium))
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(active ? Color.accentColor.opacity(0.18) : Color.primary.opacity(0.06))
            .foregroundStyle(active ? Color.accentColor : .primary)
            .clipShape(Capsule())
    }
}

struct ContentView: View {
    @EnvironmentObject var s: Session
    private let langs = ["zh", "en", "vi", "th", "lo", "km", "my", "ja", "ko", "de", "fr", "es"]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                transcript
                Divider()
                sidebar.frame(width: 260)
            }
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: header — connection + routes

    private var header: some View {
        HStack(spacing: 12) {
            Text("VoxWeft").font(.system(size: 20, weight: .semibold))
            Circle().fill(s.connected ? Color.green : Color.secondary).frame(width: 9, height: 9)
            Text(s.connected ? "local models connected" : s.status)
                .font(.system(size: 13)).foregroundStyle(.secondary)
            Spacer()
            TextField("ws://host:3100/ws/voice/v2", text: $s.serverURL)
                .textFieldStyle(.roundedBorder).frame(width: 300).disabled(s.connected)
            Chip(text: s.connected ? "Disconnect" : "Connect", active: !s.connected)
                .tappable { s.connected ? s.disconnect() : s.connect() }
        }
        .padding(14)
    }

    // MARK: transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(s.turns) { t in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(t.src.isEmpty ? "…" : t.src)
                                .font(.system(size: 17)).foregroundStyle(.secondary)
                            Text(t.out.isEmpty ? " " : t.out)
                                .font(.system(size: 22, weight: .medium))
                            HStack(spacing: 10) {
                                if let a = t.asrMs { Chip(text: "ASR \(a) ms") }
                                if let m = t.mtMs { Chip(text: "MT \(m) ms") }
                                if let f = t.firstAudioMs { Chip(text: "first audio \(f) ms", active: true) }
                                if let tot = t.totalMs { Chip(text: "total \(tot) ms") }
                            }
                        }
                        .id(t.id)
                        .padding(.horizontal, 20)
                    }
                }
                .padding(.vertical, 16)
            }
            .onChange(of: s.turns.count) { _, _ in
                if let last = s.turns.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    // MARK: sidebar — mode, languages, levels

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 18) {
            Group {
                Text("Mode").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Chip(text: "Interpreter", active: s.mode == "interpreter").tappable { s.mode = "interpreter" }
                    Chip(text: "Dialogue", active: s.mode == "agent").tappable { s.mode = "agent" }
                }
            }
            Group {
                Text("Source → Target").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Picker("", selection: $s.srcLang) { ForEach(langs, id: \.self) { Text($0) } }
                        .labelsHidden().frame(width: 90)
                    Text("→")
                    Picker("", selection: $s.tgtLang) { ForEach(langs, id: \.self) { Text($0) } }
                        .labelsHidden().frame(width: 90)
                }
                if !s.routes.isEmpty {
                    Text(s.routes).font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            Group {
                Text("Levels").font(.caption).foregroundStyle(.secondary)
                Level(label: "mic", value: s.micLevel, tint: .blue)
                Level(label: "speaker", value: s.speakerLevel, tint: .green)
            }
            Group {
                Text("Mic-free demo").font(.caption).foregroundStyle(.secondary)
                Chip(text: "▶︎ Play repo intro (\(s.srcLang))", active: true).tappable { s.playIntro() }
                Chip(text: "■ Stop").tappable { s.stopIntro(); s.interrupt() }
            }
            Spacer()
            if !s.lastError.isEmpty {
                Text(s.lastError).font(.system(size: 12)).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
    }

    // MARK: footer — hold to talk

    private var footer: some View {
        HStack {
            Spacer()
            Text(s.recording ? "Listening… release to translate" : "Hold to talk")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 320, height: 56)
                .background(s.recording ? Color.red : (s.connected ? Color.accentColor : Color.gray))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in if !s.recording { s.pttDown() } }
                        .onEnded { _ in s.pttUp() }
                )
            Spacer()
            Chip(text: "Interrupt").tappable { s.interrupt() }
            Chip(text: "Clear").tappable { s.turns.removeAll() }
        }
        .padding(14)
    }
}

private struct Level: View {
    let label: String
    let value: Float
    let tint: Color
    var body: some View {
        HStack {
            Text(label).font(.system(size: 11, design: .monospaced)).frame(width: 56, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule().fill(tint).frame(width: g.size.width * CGFloat(min(1, value * 6)))
                }
            }
            .frame(height: 8)
        }
    }
}
