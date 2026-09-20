import SwiftUI

@main
struct VoxWeftApp: App {
    @StateObject private var session = Session()

    var body: some Scene {
        WindowGroup("VoxWeft") {
            ContentView().environmentObject(session)
                .frame(minWidth: 960, minHeight: 680)
                // The neon dashboard only reads on a dark ground, and the
                // semantic text colours have to follow it.
                .preferredColorScheme(.dark)
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

// MARK: - palette — speed reads as teal on near-black, everything glows

enum Palette {
    static let warp = Color(red: 0.31, green: 0.84, blue: 1.00)      // #4FD6FF  source side
    static let weft = Color(red: 0.16, green: 0.96, blue: 0.76)      // #29F5C2  interpreter side
    static let spring = Color(red: 0.55, green: 0.99, blue: 0.55)    // #8CFD8C  TTS node
    static let ink = Color(red: 0.02, green: 0.05, blue: 0.06)
    static let paper = Color(red: 0.027, green: 0.055, blue: 0.063)  // #070E10
    static let panel = Color(red: 0.016, green: 0.035, blue: 0.043)
    static let card = Color(red: 0.16, green: 0.96, blue: 0.76).opacity(0.045)
    static let hair = Color(red: 0.16, green: 0.96, blue: 0.76).opacity(0.14)
}

/// Layered blur behind a shape — one glow helper so the look stays consistent.
extension View {
    func glow(_ color: Color, radius: CGFloat = 12, opacity: Double = 0.55) -> some View {
        self
            .shadow(color: color.opacity(opacity), radius: radius * 0.4)
            .shadow(color: color.opacity(opacity * 0.6), radius: radius)
            .shadow(color: color.opacity(opacity * 0.3), radius: radius * 2)
    }
}

struct Pill: View {
    let text: String
    var tint: Color = .secondary
    var filled = false
    var mono = false
    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .medium, design: mono ? .monospaced : .default))
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(
                Capsule().fill(filled ? tint.opacity(0.18) : Palette.card)
                    .overlay(Capsule().stroke(filled ? tint.opacity(0.5) : Palette.hair, lineWidth: 1))
            )
            .foregroundStyle(filled ? tint : .secondary)
            .clipShape(Capsule())
            .glow(filled ? tint : .clear, radius: 8, opacity: filled ? 0.4 : 0)
    }
}

struct Logo: View {
    var size: CGFloat = 28
    var body: some View {
        Group {
            if let img = Self.image { Image(nsImage: img).resizable().interpolation(.high) }
            else { RoundedRectangle(cornerRadius: size * 0.22).fill(Palette.ink) }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.22))
    }
    static let image: NSImage? = {
        for name in ["logo@2x", "logo"] {
            if let u = Bundle.main.url(forResource: name, withExtension: "png"), let i = NSImage(contentsOf: u) { return i }
            if let exe = Bundle.main.executableURL {
                let u = exe.deletingLastPathComponent().appendingPathComponent("../Resources/\(name).png").standardized
                if let i = NSImage(contentsOf: u) { return i }
            }
        }
        return nil
    }()
}

// MARK: - pipeline dashboard

/// ASR → LLM → TTS with the working node breathing. The stage comes straight
/// off the session, so what glows is what the machine is doing right now —
/// that is the whole point of showing it during a demo.
struct PipelineDashboard: View {
    let stage: Session.Stage
    let connected: Bool

    private struct Node {
        let label: String
        let sub: String
        let stage: Session.Stage
        let tint: Color
    }

    private let nodes = [
        Node(label: "ASR", sub: "listen", stage: .asr, tint: Palette.warp),
        Node(label: "LLM", sub: "translate", stage: .llm, tint: Palette.weft),
        Node(label: "TTS", sub: "speak", stage: .tts, tint: Palette.spring),
    ]

    var body: some View {
        // 30 fps is plenty for a breath and leaves the GPU to the models.
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { ctx in
            let t = ctx.date.timeIntervalSinceReferenceDate
            let breath = 0.5 + 0.5 * sin(t * 3.0)           // ~2 s in and out
            let standby = 0.5 + 0.5 * sin(t * 0.9)          // slow idle shimmer

            VStack(spacing: 9) {
                HStack(spacing: 0) {
                    ForEach(Array(nodes.enumerated()), id: \.offset) { i, n in
                        if i > 0 {
                            connector(lit: stage > nodes[i - 1].stage,
                                      flowing: stage == n.stage, t: t, tint: n.tint)
                        }
                        node(n, breath: breath, standby: standby)
                    }
                }
                Text("on device inference")
                    .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                    .kerning(3.6)
                    .foregroundStyle(Palette.weft.opacity(connected ? 0.8 : 0.35))
                    .glow(connected ? Palette.weft : .clear, radius: 7, opacity: 0.45)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(dashboardGround(glow: standby))
        }
    }

    private func dashboardGround(glow: Double) -> some View {
        ZStack {
            Palette.panel
            RadialGradient(colors: [Palette.weft.opacity(0.10 + 0.05 * glow), .clear],
                           center: .center, startRadius: 1, endRadius: 420)
        }
        .ignoresSafeArea(edges: .horizontal)
    }

    private func node(_ n: Node, breath: Double, standby: Double) -> some View {
        let active = stage == n.stage
        let done = stage > n.stage
        let lit = active || done
        let pulse = active ? breath : (connected && stage <= .listening ? standby * 0.35 : 0)

        return VStack(spacing: 2) {
            Text(n.label)
                .font(.system(size: 18, weight: .bold, design: .rounded)).kerning(1.6)
            Text(n.sub)
                .font(.system(size: 8.5, weight: .medium, design: .monospaced)).kerning(1.2)
                .opacity(0.8)
        }
        .foregroundStyle(lit ? n.tint : n.tint.opacity(0.3 + 0.25 * pulse))
        .frame(width: 118, height: 54)
        .background(
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .fill(n.tint.opacity(lit ? 0.09 + 0.10 * pulse : 0.025 + 0.02 * pulse))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .stroke(lit ? n.tint.opacity(0.4 + 0.5 * pulse)
                            : n.tint.opacity(0.14 + 0.12 * pulse),
                        lineWidth: active ? 1.6 : 1)
        )
        .glow(lit ? n.tint : .clear,
              radius: active ? 11 + 13 * pulse : (done ? 7 : 0),
              opacity: active ? 0.45 + 0.35 * pulse : (done ? 0.25 : 0))
        .scaleEffect(1 + 0.025 * (active ? breath : 0))
    }

    /// The dot rides the wire into whichever node is currently working.
    private func connector(lit: Bool, flowing: Bool, t: Double, tint: Color) -> some View {
        let w: CGFloat = 58
        let p = CGFloat((t * 0.85).truncatingRemainder(dividingBy: 1.0))
        return ZStack(alignment: .leading) {
            Capsule()
                .fill(lit ? tint.opacity(0.4) : Palette.hair)
                .frame(height: 2)
                .glow(lit ? tint : .clear, radius: 5, opacity: 0.4)
            if flowing {
                Circle().fill(tint).frame(width: 5, height: 5)
                    .glow(tint, radius: 7, opacity: 0.9)
                    .offset(x: (w - 5) * p)
            }
        }
        .frame(width: w, height: 6)
    }
}

// MARK: - root

struct ContentView: View {
    @EnvironmentObject var s: Session

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Palette.hair)
            PipelineDashboard(stage: s.stage, connected: s.connected)
            Divider().overlay(Palette.hair)
            HStack(spacing: 0) {
                transcript
                Divider().overlay(Palette.hair)
                sidebar.frame(width: 300)
            }
            Divider().overlay(Palette.hair)
            footer
        }
        .background(
            ZStack {
                Palette.paper
                RadialGradient(colors: [Palette.weft.opacity(0.06), .clear],
                               center: .top, startRadius: 40, endRadius: 720)
            }
        )
    }

    // MARK: header

    private var header: some View {
        HStack(spacing: 14) {
            Logo(size: 30)
            VStack(alignment: .leading, spacing: 1) {
                Text("VoxWeft").font(.system(size: 19, weight: .semibold)).kerning(0.3)
                Text("on-device simultaneous interpretation")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 8) {
                Circle().fill(s.connected ? Palette.weft : Color.secondary.opacity(0.5))
                    .frame(width: 8, height: 8)
                    .glow(s.connected ? Palette.weft : .clear, radius: 7, opacity: 0.8)
                Text(s.connected ? "local models · connected" : s.status)
                    .font(.system(size: 12)).foregroundStyle(.secondary)
            }
            TextField("ws://host:3100/ws/voice/v2", text: $s.serverURL)
                .textFieldStyle(.roundedBorder).font(.system(size: 12, design: .monospaced))
                .frame(width: 280).disabled(s.connected)
            Pill(text: s.connected ? "Disconnect" : "Connect", tint: Palette.warp, filled: !s.connected)
                .tappable { s.connected ? s.disconnect() : s.connect() }
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
    }

    // MARK: transcript — one card per turn: speaker line, interpreter line, timing ribbon

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if s.turns.isEmpty { emptyState.padding(.top, 60) }
                    ForEach(s.turns) { t in
                        TurnCard(turn: t, src: s.srcLang, tgt: s.tgtLang)
                            .id(t.id)
                    }
                }
                .padding(.horizontal, 24).padding(.vertical, 20)
            }
            .onChange(of: s.turns.count) { _, _ in
                if let last = s.turns.last { withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Logo(size: 72).opacity(0.9)
            Text("Hold the button and speak, or play the intro without a microphone.")
                .font(.system(size: 14)).foregroundStyle(.secondary)
            Text("Speech is segmented while you talk; every segment is recognized, translated and spoken on this machine.")
                .font(.system(size: 12)).foregroundStyle(.tertiary)
                .multilineTextAlignment(.center).frame(maxWidth: 460)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: sidebar

    private let langs = ["zh", "en", "de", "fr", "es", "it", "pt", "ja", "ko", "vi", "th", "id", "ms", "lo", "km", "my"]

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 22) {
            section("Mode") {
                HStack(spacing: 8) {
                    Pill(text: "Interpreter", tint: Palette.warp, filled: s.mode == "interpreter")
                        .tappable { s.mode = "interpreter"; s.turns.removeAll() }
                    Pill(text: "Dialogue", tint: Palette.weft, filled: s.mode == "agent")
                        .tappable { s.mode = "agent"; s.turns.removeAll() }
                }
            }
            section("Languages") {
                HStack(spacing: 10) {
                    langPicker($s.srcLang)
                    Image(systemName: "arrow.right").foregroundStyle(Palette.weft).font(.system(size: 13, weight: .semibold))
                    langPicker($s.tgtLang)
                }
                if !s.routes.isEmpty {
                    Text(routeSummary).font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
                }
            }
            section("Levels") {
                Level(label: "mic", value: s.micLevel, tint: Palette.warp)
                Level(label: "out", value: s.speakerLevel, tint: Palette.weft)
            }
            section("Mic-free demo") {
                Pill(text: "▶︎  Play repo intro · \(s.srcLang)", tint: Palette.weft, filled: true)
                    .tappable { s.playIntro() }
                Pill(text: "■  Stop").tappable { s.stopIntro(); s.interrupt() }
                Text("Each line is read in the source-language voice, then interpreted into the target voice.")
                    .font(.system(size: 11)).foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            if !s.audioNote.isEmpty {
                Label(s.audioNote, systemImage: "speaker.slash").font(.system(size: 11)).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !s.lastError.isEmpty {
                Label(s.lastError, systemImage: "exclamationmark.triangle").font(.system(size: 11)).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(18)
    }

    private func section<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundStyle(.tertiary)
            content()
        }
    }

    private func langPicker(_ sel: Binding<String>) -> some View {
        Picker("", selection: sel) { ForEach(langs, id: \.self) { Text($0.uppercased()).font(.system(.body, design: .monospaced)) } }
            .labelsHidden().frame(width: 84)
    }

    /// "ASR=qwen3asr MT=llm" → readable one-liner
    private var routeSummary: String {
        var parts: [String] = []
        if s.routes.contains("ASR=whisper") { parts.append("ASR whisper") } else if s.routes.contains("ASR=") { parts.append("ASR qwen3") }
        if s.routes.contains("MT=llm") { parts.append("MT llm · direct") } else if s.routes.contains("MT=nllb") { parts.append("MT dedicated") }
        return parts.joined(separator: "   ")
    }

    // MARK: footer — hold to talk

    private var footer: some View {
        HStack(spacing: 16) {
            Pill(text: "Clear").tappable { s.turns.removeAll() }
            Spacer()
            HoldToTalk(recording: s.recording, enabled: s.connected, down: s.pttDown, up: s.pttUp)
            Spacer()
            Pill(text: "Interrupt", tint: Palette.weft).tappable { s.interrupt() }
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
    }
}

// MARK: - turn card

struct TurnCard: View {
    let turn: Session.Turn
    let src: String
    let tgt: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            line(tag: src, text: turn.src.isEmpty ? "…" : turn.src, tint: Palette.warp, big: false)
            line(tag: tgt, text: turn.out.isEmpty ? " " : turn.out, tint: Palette.weft, big: true)
            ribbon
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Palette.card))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Palette.hair, lineWidth: 1))
        .glow(Palette.weft, radius: 10, opacity: 0.12)
    }

    private func line(tag: String, text: String, tint: Color, big: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(tag.uppercased())
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(tint).frame(width: 26, alignment: .leading)
            Text(text)
                .font(.system(size: big ? 22 : 15, weight: big ? .medium : .regular))
                .foregroundStyle(big ? .primary : .secondary)
                .textSelection(.enabled)
        }
    }

    private var ribbon: some View {
        HStack(spacing: 6) {
            if let a = turn.asrMs { Pill(text: "asr \(a)", mono: true) }
            if let m = turn.mtMs { Pill(text: "mt \(m)", mono: true) }
            if let f = turn.firstAudioMs {
                Pill(text: "first audio \(f) ms", tint: Palette.spring, filled: true, mono: true)
                    .help("measured from the moment VoxCPM2 received this text")
            }
            Spacer()
        }
        .padding(.leading, 38)
    }
}

// MARK: - controls

struct HoldToTalk: View {
    let recording: Bool
    let enabled: Bool
    let down: () -> Void
    let up: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: recording ? "waveform" : "mic.fill").font(.system(size: 16, weight: .semibold))
            Text(recording ? "Listening — release to interpret" : "Hold to talk").font(.system(size: 16, weight: .semibold))
        }
        .foregroundStyle(Palette.ink)
        .frame(width: 340, height: 54)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(recording
                      ? LinearGradient(colors: [Palette.spring, Palette.weft], startPoint: .leading, endPoint: .trailing)
                      : (enabled
                         ? LinearGradient(colors: [Palette.weft, Palette.warp], startPoint: .leading, endPoint: .trailing)
                         : LinearGradient(colors: [Color.gray.opacity(0.4), Color.gray.opacity(0.4)], startPoint: .leading, endPoint: .trailing)))
        )
        .glow(enabled ? (recording ? Palette.spring : Palette.weft) : .clear,
              radius: recording ? 22 : 14, opacity: recording ? 0.75 : 0.45)
        .scaleEffect(recording ? 1.02 : 1.0)
        .animation(.easeOut(duration: 0.15), value: recording)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in if !recording && enabled { down() } }
                .onEnded { _ in up() }
        )
    }
}

private struct Level: View {
    let label: String
    let value: Float
    let tint: Color
    var body: some View {
        HStack(spacing: 10) {
            Text(label).font(.system(size: 10, design: .monospaced)).foregroundStyle(.tertiary).frame(width: 26, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Palette.hair)
                    Capsule().fill(tint).frame(width: g.size.width * CGFloat(min(1, value * 6)))
                        .glow(tint, radius: 6, opacity: 0.6)
                        .animation(.linear(duration: 0.05), value: value)
                }
            }
            .frame(height: 6)
        }
    }
}
