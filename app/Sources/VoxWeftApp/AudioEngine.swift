import AVFoundation
import Accelerate

/// 全双工音频：一路采集（16kHz 单声道 PCM16），一路播放（48kHz 来自服务端）。
///
/// 为什么用 Swift 而不是网页：展厅是外放 + 麦克风同场，必须开系统级 AEC
/// （`setVoiceProcessingEnabled(true)`）——它拿播放信号做参考来消回声，
/// Chrome 对 WebAudio 的回声消除在扬声器外放时效果很差。
///
/// 采集与播放**必须走同一个 AVAudioEngine**：AEC 是引擎级的，把采集和播放
/// 拆到两个引擎上就失效了，而且更容易产生回声环路。
final class AudioEngine {
    static let micSampleRate: Double = 16000
    static let chunkSamples = 512              // 32ms @16kHz — 与服务端 FRAME_S 对齐

    private var engine = AVAudioEngine()
    private var player = AVAudioPlayerNode()
    private var converter: AVAudioConverter?

    /// 采集到的 16k 单声道 PCM16 帧（1024 字节）
    var onMicFrame: ((Data) -> Void)?
    var onMicLevel: ((Float) -> Void)?
    var onSpeakerLevel: ((Float) -> Void)?

    private(set) var isRunning = false

    private let micFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                          sampleRate: micSampleRate,
                                          channels: 1,
                                          interleaved: true)!

    /// 播放格式随服务端 `audio_start.sample_rate` 变，默认 48k。
    private var playFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                           sampleRate: 48000, channels: 1,
                                           interleaved: false)!

    // 用列表攒帧、统一切成 1024 字节再发——硬件回调给的块大小不固定，
    // 直接转发会让服务端拿到非对齐的帧长。
    private var pendingMic = Data()

    // 下行播放缓冲。
    //
    // 两件事在这里一起解决：
    //  1. WS 消息边界和服务端 200 ms 块不对齐 → 攒到整块再 scheduleBuffer，
    //     不然节点在每个碎片边界各自衔接，听感是周期性「咔」。
    //  2. 服务端产出可能**慢于实时**（M2 上 TTS RTF ≈ 1.5：每 ~300 ms 才来
    //     200 ms 音频）。逐块即到即播，播放器每播完一块就饿 ~100 ms，听感是
    //     一卡一顿的细碎断裂——不是采样率问题，是喂不上。所以开播前先攒
    //     preroll 秒，播放中队列见底就重新进入攒料状态，用一次性的启动延迟
    //     换连续播放。RTF > 1 时每句仍会有一次卡顿（攒的 preroll 用完之后），
    //     preroll 越大越少；实时机器上（M3 Ultra）preroll 只是一次 0.6 s 延迟。
    private var pendingPlay = Data()
    private let playChunkBytes = 9600 * 2      // 200ms @48kHz，与服务端一致
    private var playFlushTimer: Timer?
    /// 攒料阈值（字节）。0.6 s @48k int16。
    private let prerollBytes = 48000 * 2 * 6 / 10
    /// true = 正在攒料，攒够 preroll 前不向节点排队
    private var buffering = true
    /// 已排队未播完的 buffer 数（在 completion 回调里递减）
    private var queued = 0

    // MARK: - 生命周期

    func start() throws {
        guard !isRunning else { return }
        // Voice processing (system AEC) is what makes speaker + mic in one
        // room work. It is not universally available: on a MacBook Pro whose
        // built-in array reports 3 input channels, enabling VP turns the
        // input into 7 channels and the engine's output unit then fails to
        // initialize (-10875). Try with VP first; if start() fails, rebuild
        // the engine without it. Playback is more important than AEC.
        do {
            try startEngine(voiceProcessing: true)
        } catch {
            NSLog("AudioEngine: start with voice processing failed (%@); retrying without", "\(error)")
            try startEngine(voiceProcessing: false)
        }
    }

    private(set) var voiceProcessing = false

    private func startEngine(voiceProcessing vp: Bool) throws {
        // A fresh engine each attempt: setVoiceProcessingEnabled cannot be
        // toggled back reliably on a configured graph.
        engine = AVAudioEngine()
        player = AVAudioPlayerNode()
        let input = engine.inputNode

        if vp { try input.setVoiceProcessingEnabled(true) }
        let hwFormat = input.outputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0, hwFormat.channelCount > 0 else {
            throw AudioError.noInputDevice
        }
        guard let conv = AVAudioConverter(from: hwFormat, to: micFormat) else {
            throw AudioError.converterUnavailable
        }
        converter = conv

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: playFormat)

        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buf, _ in
            self?.handleMic(buf)
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        player.play()
        voiceProcessing = vp
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        player.stop()
        engine.stop()
        engine.disconnectNodeOutput(player)
        isRunning = false
        pendingMic.removeAll()
    }

    // MARK: - 采集

    private func handleMic(_ buffer: AVAudioPCMBuffer) {
        guard let conv = converter else { return }

        let ratio = micFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: micFormat, frameCapacity: capacity) else { return }

        var err: NSError?
        var consumed = false
        conv.convert(to: out, error: &err) { _, status in
            // 每块输入只喂一次；再要就报 noData，否则 converter 会重复消费
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard err == nil, out.frameLength > 0, let ptr = out.int16ChannelData?[0] else { return }

        let count = Int(out.frameLength)
        let data = Data(bytes: ptr, count: count * 2)

        // 电平表（用采集后的 16k 数据算，和送给服务端的是同一份）
        onMicLevel?(Self.rms(ptr, count))

        pendingMic.append(data)
        let frameBytes = Self.chunkSamples * 2      // 1024
        while pendingMic.count >= frameBytes {
            let frame = pendingMic.prefix(frameBytes)
            pendingMic.removeFirst(frameBytes)
            onMicFrame?(Data(frame))
        }
    }

    // MARK: - 播放

    /// 播放服务端下行的一块 PCM16。`sampleRate` 变了会自动重建连接。
    func play(_ data: Data, sampleRate: Double) {
        guard isRunning, !data.isEmpty else { return }
        if abs(playFormat.sampleRate - sampleRate) > 1 {
            rebuildPlayback(sampleRate: sampleRate)
        }
        pendingPlay.append(data)
        if buffering && pendingPlay.count < prerollBytes {
            scheduleFlushTimer()           // 句子比 preroll 还短时兜底
            return
        }
        buffering = false
        drainPlaybackBuffer()
        if !pendingPlay.isEmpty { scheduleFlushTimer() }
    }

    /// 服务端说这一段结束了：把尾巴冲出去，不必等 preroll。
    func endOfUtterance() {
        playFlushTimer?.invalidate(); playFlushTimer = nil
        buffering = false
        drainPlaybackBuffer()
        if !pendingPlay.isEmpty {
            let rest = pendingPlay; pendingPlay.removeAll(); enqueue(rest)
        }
        buffering = true                    // 下一段重新攒料
    }

    /// 把攒够的整块交给播放节点。
    private func drainPlaybackBuffer() {
        while pendingPlay.count >= playChunkBytes {
            let chunk = pendingPlay.prefix(playChunkBytes)
            pendingPlay.removeFirst(playChunkBytes)
            enqueue(Data(chunk))
        }
    }

    /// 尾块不足一整块、且一段时间没有新数据：也播掉，别吞句尾。
    /// 间隔比服务端最慢出块间隔（M2 ~300 ms）长，避免把正在攒的料截成碎片。
    private func scheduleFlushTimer() {
        playFlushTimer?.invalidate()
        playFlushTimer = Timer.scheduledTimer(withTimeInterval: 0.5,
                                              repeats: false) { [weak self] _ in
            guard let self, !self.pendingPlay.isEmpty else { return }
            self.buffering = false
            let rest = self.pendingPlay
            self.pendingPlay.removeAll()
            self.enqueue(rest)
            self.buffering = true
        }
    }

    private func enqueue(_ data: Data) {
        guard isRunning, data.count >= 2 else { return }

        // 字节数不是偶数时最后一个字节无处安放，丢掉而不是让 vDSP 读越界
        let n = data.count / 2
        guard n > 0,
              let buf = AVAudioPCMBuffer(pcmFormat: playFormat,
                                         frameCapacity: AVAudioFrameCount(n)) else { return }
        buf.frameLength = AVAudioFrameCount(n)

        data.withUnsafeBytes { raw in
            guard let i16 = raw.baseAddress?.assumingMemoryBound(to: Int16.self) else { return }
            guard let out = buf.floatChannelData?[0] else { return }
            // vDSP 一次转完，比逐样本循环快得多（48k 流式播放时这个循环很显眼）
            var scale: Float = 1.0 / 32768.0
            vDSP_vflt16(i16, 1, out, 1, vDSP_Length(n))
            vDSP_vsmul(out, 1, &scale, out, 1, vDSP_Length(n))
        }

        if let ch = buf.floatChannelData?[0] {
            onSpeakerLevel?(Self.rms(ch, n))
        }
        queued += 1
        player.scheduleBuffer(buf) { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                self.queued -= 1
                // 队列见底 = 服务端没跟上。回到攒料状态，下一次攒够 preroll
                // 再续播，而不是每来一块播一块地断续。
                if self.queued == 0 && self.pendingPlay.count < self.prerollBytes {
                    self.buffering = true
                }
            }
        }
    }

    private func rebuildPlayback(sampleRate: Double) {
        playFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: sampleRate, channels: 1,
                                   interleaved: false)!
        engine.disconnectNodeOutput(player)
        engine.connect(player, to: engine.mainMixerNode, format: playFormat)
        player.play()
    }

    /// 打断：立刻停掉正在播的内容并清空已排队的块。
    ///
    /// `player.stop()` 会把已 schedule 但未播的 buffer 一并丢弃——这正是
    /// 打断想要的语义。但**紧接着必须 `play()`**，否则后续到达的块不会有
    /// 任何输出（节点停在 stopped 状态，scheduleBuffer 只入队不播放）。
    func stopPlayback() {
        guard isRunning else { return }
        playFlushTimer?.invalidate()
        playFlushTimer = nil
        pendingPlay.removeAll()
        buffering = true
        queued = 0
        player.stop()
        player.play()
    }

    // MARK: - 工具

    private static func rms(_ p: UnsafePointer<Int16>, _ n: Int) -> Float {
        guard n > 0 else { return 0 }
        var sum: Float = 0
        for i in 0..<n {
            let s = Float(p[i]) / 32768.0
            sum += s * s
        }
        return sqrt(sum / Float(n))
    }

    private static func rms(_ p: UnsafePointer<Float>, _ n: Int) -> Float {
        guard n > 0 else { return 0 }
        var sum: Float = 0
        for i in 0..<n { sum += p[i] * p[i] }
        return sqrt(sum / Float(n))
    }

    enum AudioError: LocalizedError {
        case noInputDevice
        case converterUnavailable

        var errorDescription: String? {
            switch self {
            case .noInputDevice:
                return "No microphone. Check System Settings › Sound › Input."
            case .converterUnavailable:
                return "Could not build audio converter (hardware sample rate 0?)."
            }
        }
    }
}
