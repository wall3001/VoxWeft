import Foundation
import Combine

/// One /ws/voice/v2 session: state the UI shows, plus the audio plumbing.
///
/// Everything the server tells us lands here as published properties;
/// the view is a dumb renderer. Latency numbers are measured on this side —
/// they are what the person in front of the machine experiences, not what
/// the server thinks it did.
@MainActor
final class Session: ObservableObject {
    // Connection
    @Published var serverURL = UserDefaults.standard.string(forKey: "serverURL")
        ?? "ws://127.0.0.1:3100/ws/voice/v2"
    @Published var connected = false
    @Published var status = "idle"

    // Config
    @Published var mode = "interpreter" { didSet { pushConfig() } }
    @Published var srcLang = "zh" { didSet { pushConfig() } }
    @Published var tgtLang = "en" { didSet { pushConfig() } }
    @Published var routes = ""

    // Transcript
    struct Turn: Identifiable {
        let id = UUID()
        var seq: Int = 0          // server segment seq; 0 = text_input turn
        var src = ""
        var out = ""
        var firstAudioMs: Int?
        var totalMs: Int?
        var asrMs: Int?
        var mtMs: Int?
    }
    @Published var turns: [Turn] = []

    // Live levels
    @Published var micLevel: Float = 0
    @Published var speakerLevel: Float = 0
    @Published var recording = false
    @Published var lastError = ""
    @Published var audioNote = ""

    private let ws = WebSocketManager()
    private let audio = AudioEngine()
    private var turnStart: Date?
    private var configDirty = false
    private var sourcePlaying = false
    private var demoQueue: [String] = []
    private var demoTimer: Timer?

    init() {
        ws.onConnected = { [weak self] in
            Task { @MainActor in self?.connected = true; self?.status = "connected" }
        }
        ws.onDisconnected = { [weak self] why in
            Task { @MainActor in
                self?.connected = false; self?.status = "disconnected: \(why)"
            }
        }
        ws.onJSON = { [weak self] obj in
            Task { @MainActor in self?.handle(obj) }
        }
        ws.onAudio = { [weak self] data in
            Task { @MainActor in self?.handleAudio(data) }
        }
        audio.onMicFrame = { [weak self] frame in self?.ws.sendAudio(frame) }
        audio.onMicLevel = { [weak self] lvl in
            Task { @MainActor in self?.micLevel = lvl }
        }
        audio.onSpeakerLevel = { [weak self] lvl in
            Task { @MainActor in self?.speakerLevel = lvl }
        }
    }

    // MARK: connection

    func connect() {
        UserDefaults.standard.set(serverURL, forKey: "serverURL")
        status = "connecting…"
        ws.connect(to: serverURL)
        do {
            try audio.start()
            audioNote = audio.voiceProcessing ? "" : "echo cancellation off (mic array unsupported) — use headphones or keep the volume low"
        } catch {
            lastError = "audio: \(error.localizedDescription)"
        }
    }

    func disconnect() {
        ws.disconnect(); audio.stop(); connected = false; status = "idle"
    }

    private func pushConfig() {
        guard connected else { return }
        // Never reconfigure mid-press: the server would rebuild the
        // orchestrator under audio already in flight (DEVLOG §4.7).
        // Remember and send on release instead of dropping the change.
        guard !recording else { configDirty = true; return }
        configDirty = false
        ws.sendJSON(["type": "session_config", "mode": mode,
                     "src_lang": srcLang, "tgt_lang": tgtLang])
    }

    // MARK: push-to-talk

    func pttDown() {
        guard connected, !recording else { return }
        audio.stopPlayback()
        recording = true
        ws.sendJSON(["type": "ptt_start"])
    }

    func pttUp() {
        guard recording else { return }
        recording = false
        ws.sendJSON(["type": "ptt_end"])
        // Latency clock starts at release: that is when the user is waiting.
        turnStart = Date()
        if configDirty { pushConfig() }
    }

    func interrupt() {
        audio.stopPlayback()
        ws.sendJSON(["type": "interrupt", "reason": "user"])
    }

    // MARK: mic-free demo — feeds the bundled intro text as text_input

    func playIntro() {
        guard connected, demoQueue.isEmpty else { return }
        guard let text = Self.introText(lang: srcLang) else {
            lastError = "no bundled intro for \(srcLang)"; return
        }
        demoQueue = text.split(separator: "\n").map(String.init)
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        status = "demo: \(demoQueue.count) sentences"
        sendNextDemoLine()
    }

    func stopIntro() {
        demoQueue.removeAll(); demoTimer?.invalidate(); demoTimer = nil
        sourcePlaying = false
        interrupt()
        status = "connected"
    }

    private func sendNextDemoLine() {
        guard !demoQueue.isEmpty else { status = "connected"; return }
        let line = demoQueue.removeFirst()
        turnStart = Date()
        turns.append(Turn(src: line))
        ws.sendJSON(["type": "text_input", "text": line])
    }

    // MARK: server events

    private func handle(_ obj: [String: Any]) {
        guard let type = obj["type"] as? String else { return }
        switch type {
        case "ready":
            pushConfig()
        case "session_config_ack":
            routes = obj["routes"] as? String ?? ""
        case "segment_committed":
            if turnStart == nil { turnStart = Date() }
            turns.append(Turn(seq: (obj["seq"] as? NSNumber)?.intValue ?? 0))
        case "asr":
            let i = turnIndex(for: obj)
            turns[i].src = obj["text"] as? String ?? ""
            turns[i].asrMs = (obj["latency_ms"] as? NSNumber)?.intValue
        case "translation", "llm":
            let i = turnIndex(for: obj)
            turns[i].out = obj["text"] as? String ?? ""
            turns[i].mtMs = (obj["latency_ms"] as? NSNumber)?.intValue
        case "llm_sentence":
            let i = turnIndex(for: obj)
            turns[i].out += obj["text"] as? String ?? ""
        case "source_audio":
            // Demo: the source line is being read aloud first. Restart the
            // latency clock when it finishes so "first audio" measures the
            // interpretation, not the read-out.
            sourcePlaying = true
        case "audio_end":
            audio.endOfUtterance()
            if sourcePlaying { sourcePlaying = false; turnStart = Date() }
        case "audio_start":
            if sourcePlaying { break }
            let i = turnIndex(for: obj)
            if let t = turnStart, turns[i].firstAudioMs == nil {
                turns[i].firstAudioMs = Int(Date().timeIntervalSince(t) * 1000)
            }
        case "turn_end":
            if let t = turnStart, !turns.isEmpty {
                turns[turns.count - 1].totalMs = Int(Date().timeIntervalSince(t) * 1000)
            }
            turnStart = nil
            if !demoQueue.isEmpty {
                // Give playback a beat to drain before the next line.
                demoTimer = Timer.scheduledTimer(withTimeInterval: 0.6, repeats: false) { [weak self] _ in
                    Task { @MainActor in self?.sendNextDemoLine() }
                }
            }
        case "no_speech":
            let quiet = (obj["too_quiet"] as? Bool) ?? false
            lastError = quiet ? "too quiet (peak \(obj["peak_rms"] ?? "?") < \(obj["threshold"] ?? "?"))"
                              : "no speech detected"
        case "interrupted":
            audio.stopPlayback()
        case "error":
            lastError = "\(obj["stage"] ?? ""): \(obj["message"] ?? "")"
        default:
            break
        }
    }

    /// Events carry the server's segment `seq`; match on it. `seq` 0 (or
    /// absent) is a text_input turn, which the client created itself — use
    /// the last turn. Segments are committed synchronously while their ASR
    /// arrives later, so "the last turn" is wrong as soon as two segments
    /// are in flight.
    private func turnIndex(for obj: [String: Any]) -> Int {
        let seq = (obj["seq"] as? NSNumber)?.intValue ?? 0
        if seq > 0, let i = turns.lastIndex(where: { $0.seq == seq }) { return i }
        if turns.isEmpty { turns.append(Turn(seq: seq)) }
        return turns.count - 1
    }

    private func handleAudio(_ data: Data) {
        if !sourcePlaying, let t = turnStart, !turns.isEmpty, turns[turns.count - 1].firstAudioMs == nil {
            turns[turns.count - 1].firstAudioMs = Int(Date().timeIntervalSince(t) * 1000)
        }
        audio.play(data, sampleRate: 48000)
    }

    /// Bundled intro text. Under SwiftPM it is a package resource; when built
    /// with plain `swiftc` (app/build.sh, for machines with only Command Line
    /// Tools) it sits in an `intro/` directory next to the executable.
    private static func introText(lang: String) -> String? {
        let name = "intro_\(lang)"
        var candidates: [URL] = []
        #if SWIFT_PACKAGE
        if let u = Bundle.module.url(forResource: name, withExtension: "txt", subdirectory: "intro") {
            candidates.append(u)
        }
        #endif
        if let exe = Bundle.main.executableURL {
            candidates.append(exe.deletingLastPathComponent().appendingPathComponent("intro/\(name).txt"))
            candidates.append(exe.deletingLastPathComponent()
                .appendingPathComponent("../Resources/intro/\(name).txt").standardized)
        }
        for u in candidates {
            if let t = try? String(contentsOf: u, encoding: .utf8) { return t }
        }
        return nil
    }
}
