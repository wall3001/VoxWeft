import Foundation

/// WebSocket client for /ws/voice/v2.
/// Binary messages = PCM audio, text messages = JSON control.
final class WebSocketManager: NSObject, URLSessionWebSocketDelegate {
    private var task: URLSessionWebSocketTask?
    private var session: URLSession?
    private var pingTimer: Timer?

    var onConnected: (() -> Void)?
    var onDisconnected: ((String) -> Void)?
    var onAudio: ((Data) -> Void)?
    var onJSON: (([String: Any]) -> Void)?

    var isConnected: Bool { task?.state == .running }

    func connect(to urlString: String) {
        disconnect()

        guard let url = URL(string: urlString) else {
            onDisconnected?("Invalid URL")
            return
        }

        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        task = session?.webSocketTask(with: url)
        task?.resume()
    }

    func disconnect() {
        pingTimer?.invalidate()
        pingTimer = nil
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        session?.invalidateAndCancel()
        session = nil
    }

    /// 已发送的帧计数与失败计数，供排查「点了没反应」这类问题。
    /// 之前这里的回调是 `{ _ in }`，把所有发送错误都吞掉了。
    private(set) var audioFramesSent = 0
    private(set) var audioSendErrors = 0

    func sendAudio(_ data: Data) {
        guard let task, task.state == .running else {
            audioSendErrors += 1
            return
        }
        audioFramesSent += 1
        task.send(.data(data)) { [weak self] err in
            if err != nil { self?.audioSendErrors += 1 }
        }
    }

    func sendJSON(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { _ in }
    }

    // MARK: - URLSessionWebSocketDelegate

    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol proto: String?) {
        onConnected?()
        startReceiving()
        startPing()
    }

    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
                    reason: Data?) {
        let msg = reason.flatMap { String(data: $0, encoding: .utf8) } ?? "closed"
        onDisconnected?(msg)
    }

    // MARK: - Internal

    /// 接收循环。**这是唯一一次性的、不重试的循环**——一旦中断就再也收不到
    /// 任何东西，而且不会产生任何错误信号（表现为「服务端明明在发，客户端
    /// 一点反应没有」，极易误判成服务端问题）。
    ///
    /// 用**本轮拿到的 task** 继续收，而不是 `self.task`：若期间重连过，
    /// `self.task` 已指向新连接，旧循环会静默停住。
    private func startReceiving() {
        let t = task
        t?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                switch message {
                case .data(let data):
                    self.onAudio?(data)
                case .string(let text):
                    if let data = text.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        self.onJSON?(json)
                    }
                @unknown default:
                    break
                }
                if let t, t === self.task {
                    self.startReceiving()
                }
            case .failure:
                self.onDisconnected?("Connection lost")
            }
        }
    }

    private func startPing() {
        DispatchQueue.main.async { [weak self] in
            self?.pingTimer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in
                self?.task?.sendPing { error in
                    if error != nil {
                        self?.onDisconnected?("Ping failed")
                    }
                }
            }
        }
    }
}
