import Dispatch
import Synchronization

/// Runs Vision work one body at a time, off Swift concurrency's cooperative
/// thread pool.
///
/// `VNImageRequestHandler.perform(_:)` blocks its thread while Vision works,
/// and Vision needs a free pool thread to finish that work. When every pool
/// thread is blocked in Vision, or blocked waiting to start a Vision call, the
/// process stops making progress. A conforming serializer must make waiting
/// callers suspend (not block a thread) and must run `body` outside the pool.
///
/// Hosts that make Vision requests in other packages pass one serializer to
/// every `VisionFeaturePrintBackend`, so the whole process shares one queue.
///
/// Requirements for every implementation:
/// - If the calling task is cancelled before `body` starts, skip `body` and
///   throw `CancellationError`. Once `body` has started, let it run to the end.
/// - `body` builds its Vision requests and handlers itself and returns only
///   `Sendable` values. It runs outside any task, so `Task.isCancelled` is
///   always `false` inside it.
/// - Never call `run` from inside a body: the inner call would wait for the
///   outer one, which never finishes.
public protocol VisionWorkSerializing: Sendable {
    func run<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T
}

/// The default `VisionWorkSerializing`: one private serial `DispatchQueue`.
public final class SerialVisionWorkQueue: VisionWorkSerializing {
    /// Shared by every `VisionFeaturePrintBackend` that is not given its own
    /// serializer.
    public static let shared = SerialVisionWorkQueue(label: "PhotoAIKit.SerialVisionWorkQueue")

    private let queue: DispatchQueue

    public init(label: String) {
        queue = DispatchQueue(label: label, qos: .userInitiated, autoreleaseFrequency: .workItem)
    }

    public func run<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        let ticket = Ticket<T>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                guard ticket.wait(with: continuation) else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                queue.async {
                    guard let continuation = ticket.start() else { return }
                    continuation.resume(with: Result { try body() })
                }
            }
        } onCancel: {
            ticket.cancel()?.resume(throwing: CancellationError())
        }
    }
}

/// One caller's place in the queue. Whoever moves it out of `.waiting`
/// (the queue starting the body, or cancellation) resumes the continuation.
private final class Ticket<T: Sendable>: Sendable {
    private enum State {
        case idle
        case waiting(CheckedContinuation<T, any Error>)
        case started
        case cancelled
    }

    private let state = Mutex(State.idle)

    /// Stores the continuation. Returns `false` if the caller was already cancelled.
    func wait(with continuation: CheckedContinuation<T, any Error>) -> Bool {
        state.withLock { state in
            guard case .idle = state else { return false }
            state = .waiting(continuation)
            return true
        }
    }

    /// Called on the queue. Returns the continuation if the body should run.
    func start() -> CheckedContinuation<T, any Error>? {
        state.withLock { state in
            guard case let .waiting(continuation) = state else { return nil }
            state = .started
            return continuation
        }
    }

    /// Returns the continuation if cancellation won the race with `start()`.
    func cancel() -> CheckedContinuation<T, any Error>? {
        state.withLock { state in
            switch state {
            case .idle:
                state = .cancelled
                return nil
            case let .waiting(continuation):
                state = .cancelled
                return continuation
            case .started, .cancelled:
                return nil
            }
        }
    }
}
