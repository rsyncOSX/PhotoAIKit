import CoreGraphics
import Dispatch
import Foundation
import PhotoAIContracts
import Synchronization
import Testing
import VisionFeaturePrintBackend

@Suite("Vision work serializing")
struct VisionWorkSerializingTests {
    @Test("Concurrent callers never overlap")
    func bodiesNeverOverlap() async throws {
        let queue = SerialVisionWorkQueue(label: "test.overlap")
        let counts = Mutex((inFlight: 0, peak: 0))
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for index in 0..<32 {
                group.addTask {
                    try await queue.run {
                        counts.withLock {
                            $0.inFlight += 1
                            $0.peak = max($0.peak, $0.inFlight)
                        }
                        Thread.sleep(forTimeInterval: 0.002)
                        counts.withLock { $0.inFlight -= 1 }
                        return index
                    }
                }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        #expect(results.sorted() == Array(0..<32))
        #expect(counts.withLock { $0.peak } == 1)
    }

    @Test("Errors thrown by a body reach the caller")
    func bodyErrorPropagates() async {
        struct Failure: Error {}
        let queue = SerialVisionWorkQueue(label: "test.error")
        await #expect(throws: Failure.self) {
            try await queue.run { throw Failure() }
        }
    }

    @Test("Cancellation before the turn throws CancellationError and skips the body")
    func cancellationBeforeTurn() async throws {
        let queue = SerialVisionWorkQueue(label: "test.cancel")
        let release = DispatchSemaphore(value: 0)
        let blockerStarted = DispatchSemaphore(value: 0)
        let skippedBodyRan = Mutex(false)

        let blocker = Task {
            try await queue.run {
                blockerStarted.signal()
                release.wait()
            }
        }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                blockerStarted.wait()
                continuation.resume()
            }
        }

        let waiter = Task {
            try await queue.run { skippedBodyRan.withLock { $0 = true } }
        }
        waiter.cancel()
        await #expect(throws: CancellationError.self) {
            try await waiter.value
        }

        release.signal()
        try await blocker.value
        // This body runs after the cancelled ticket was drained, so it proves
        // the cancelled body was skipped.
        try await queue.run {}
        #expect(skippedBodyRan.withLock { $0 } == false)
    }

    @Test("A task cancelled before calling run never runs the body")
    func alreadyCancelled() async {
        let queue = SerialVisionWorkQueue(label: "test.precancelled")
        let ran = Mutex(false)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await queue.run { ran.withLock { $0 = true } }
        }
        await #expect(throws: CancellationError.self) {
            try await task.value
        }
        #expect(ran.withLock { $0 } == false)
    }

    @Test("The backend runs its Vision work through the injected serializer")
    func backendUsesInjectedSerializer() async throws {
        let image = try #require(makeCheckerboard(size: 128))
        let serializer = CountingSerializer()
        let backend = VisionFeaturePrintBackend(serializer: serializer)
        let first = try await backend.artifact(for: image, source: testSource())
        let second = try await backend.artifact(for: image, source: testSource())
        #expect(serializer.calls == 2)

        let distance = try #require(try backend.distance(from: first, to: second))
        #expect(distance < 0.001)
    }

    @Test("Many concurrent feature prints finish without exhausting the thread pool",
          .timeLimit(.minutes(1)))
    func manyConcurrentFeaturePrintsFinish() async throws {
        let image = try #require(makeCheckerboard(size: 256))
        let count = ProcessInfo.processInfo.activeProcessorCount * 3
        let finished = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<count {
                group.addTask {
                    // A new backend per call: the default shared queue, not
                    // actor isolation, is what keeps them in line.
                    _ = try await VisionFeaturePrintBackend().artifact(for: image, source: testSource())
                    return 1
                }
            }
            return try await group.reduce(0, +)
        }
        #expect(finished == count)
    }
}

private final class CountingSerializer: VisionWorkSerializing {
    private let count = Mutex(0)
    private let queue = SerialVisionWorkQueue(label: "test.counting")

    var calls: Int { count.withLock { $0 } }

    func run<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        count.withLock { $0 += 1 }
        return try await queue.run(body)
    }
}

private func testSource() -> AIImageSource {
    AIImageSource(id: UUID(), url: URL(fileURLWithPath: "/tmp/checkerboard.png"), displayName: "checkerboard.png")
}

private func makeCheckerboard(size: Int) -> CGImage? {
    guard let context = CGContext(
        data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    context.setFillColor(CGColor(gray: 0.08, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: size, height: size))
    context.setFillColor(CGColor(gray: 0.92, alpha: 1))
    let tile = 8
    for y in stride(from: 0, to: size, by: tile) {
        for x in stride(from: 0, to: size, by: tile) where (x / tile + y / tile).isMultiple(of: 2) {
            context.fill(CGRect(x: x, y: y, width: tile, height: tile))
        }
    }
    return context.makeImage()
}
