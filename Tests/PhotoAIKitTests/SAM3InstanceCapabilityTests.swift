import CoreAIImageSegmenter
@testable import CoreAISAM3Backend
import Darwin
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

/// Opt-in Phase 0 experiment. No model or photographs are stored in the repo.
@Suite("SAM 3 instance capability")
struct SAM3InstanceCapabilityTests {
    @Test(
        "Record real instance output for the distributed model",
        .enabled(if: DiagnosticConfiguration.isAvailable)
    )
    func recordInstanceOutput() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let bundlePath = environment["SAM3_DIAGNOSTIC_BUNDLE"],
              let manifestPath = environment["SAM3_DIAGNOSTIC_MANIFEST"],
              let outputPath = environment["SAM3_DIAGNOSTIC_OUTPUT"]
        else { return }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let outputURL = URL(fileURLWithPath: outputPath, isDirectory: true).standardizedFileURL
        let cases = try JSONDecoder().decode(
            [PhotoCase].self,
            from: Data(contentsOf: manifestURL)
        )
        try #require(!cases.isEmpty)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let provider = try CoreAISAM3Provider(
            modelBundleURL: URL(fileURLWithPath: bundlePath, isDirectory: true)
        )
        var runs: [Run] = []
        let expectedRunCount = cases.reduce(0) { $0 + $1.concepts.count * 8 }
        for photoCase in cases {
            try #require(!photoCase.id.isEmpty && !photoCase.concepts.isEmpty)
            let imageURL = URL(
                fileURLWithPath: photoCase.image,
                relativeTo: manifestURL.deletingLastPathComponent()
            ).standardizedFileURL
            let image = try loadImage(at: imageURL)
            for concept in photoCase.concepts {
                try #require(!concept.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                for cap in [5, 8, 12, 16] {
                    for repetition in 1...2 {
                        try Task.checkCancellation()
                        let measurement = try await provider.diagnoseInstances(
                            image: image,
                            concept: concept,
                            maximumSegmentCount: cap
                        )
                        let response = measurement.response
                        #expect(response.segments.count <= cap)
                        var segments: [SegmentRecord] = []
                        for (index, segment) in response.segments.enumerated() {
                            let maskName: String?
                            if repetition == 1 {
                                let name = "\(safeName(photoCase.id))-\(safeName(concept))-cap\(cap)-instance\(index + 1).png"
                                try savePNG(maskImage(for: segment), at: outputURL.appendingPathComponent(name))
                                maskName = name
                            } else {
                                maskName = nil
                            }
                            segments.append(SegmentRecord(segment, maskFile: maskName))
                        }
                        let overlayName: String?
                        if repetition == 1, !response.segments.isEmpty,
                           let overlay = SegmentationVisualization.renderInstanceMasks(
                               onto: image,
                               segments: response.segments
                           ) {
                            let name = "\(safeName(photoCase.id))-\(safeName(concept))-cap\(cap).png"
                            try savePNG(overlay, at: outputURL.appendingPathComponent(name))
                            overlayName = name
                        } else {
                            overlayName = nil
                        }
                        runs.append(Run(
                            photoID: photoCase.id,
                            imagePath: imageURL.path,
                            imageWidth: image.width,
                            imageHeight: image.height,
                            concept: concept,
                            maximumSegmentCount: cap,
                            repetition: repetition,
                            milliseconds: measurement.milliseconds,
                            peakResidentBytes: peakResidentBytes(),
                            probabilityMapPresent: response.probabilityMap != nil,
                            segmentCount: response.segments.count,
                            segments: segments,
                            overlay: overlayName
                        ))
                        try saveReport(
                            provider: provider,
                            runs: runs,
                            expectedRunCount: expectedRunCount,
                            at: outputURL
                        )
                        print("SAM3 diagnostic \(runs.count)/\(expectedRunCount): \(photoCase.id), \(concept), cap \(cap), repeat \(repetition): \(response.segments.count) segments")
                    }
                }
            }
        }

    }

    private func saveReport(
        provider: CoreAISAM3Provider,
        runs: [Run],
        expectedRunCount: Int,
        at outputURL: URL
    ) throws {
        let report = Report(
            modelIdentity: provider.modelIdentity.artifactIdentifier,
            modelAsset: provider.modelIdentity.assetName,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            inputMaxSide: 4_320,
            boxCoordinateSystem: "input-image pixels; bottom-left origin on macOS",
            expectedRunCount: expectedRunCount,
            complete: runs.count == expectedRunCount,
            runs: runs
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: outputURL.appendingPathComponent("sam3-instance-report.json"),
            options: .atomic
        )
    }

    private func loadImage(at url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: 4_320,
              ] as CFDictionary)
        else { throw DiagnosticError.cannotDecode(url.path) }
        return image
    }

    private func savePNG(_ image: CGImage, at url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { throw DiagnosticError.cannotWrite(url.path) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw DiagnosticError.cannotWrite(url.path)
        }
    }

    private func maskImage(for segment: Segment) throws -> CGImage {
        let pixels = Data(segment.mask.map { $0 ? UInt8(255) : UInt8(0) })
        guard segment.maskWidth > 0,
              segment.maskHeight > 0,
              pixels.count == segment.maskWidth * segment.maskHeight,
              let provider = CGDataProvider(data: pixels as CFData),
              let image = CGImage(
                  width: segment.maskWidth,
                  height: segment.maskHeight,
                  bitsPerComponent: 8,
                  bitsPerPixel: 8,
                  bytesPerRow: segment.maskWidth,
                  space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              )
        else { throw DiagnosticError.invalidMask }
        return image
    }

    private func peakResidentBytes() -> Int64 {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return -1 }
        return Int64(usage.ru_maxrss)
    }

    private func safeName(_ value: String) -> String {
        let cleaned = value.lowercased().map { character in
            character.isASCII && (character.isLetter || character.isNumber) ? character : "-"
        }
        return String(cleaned.prefix(60))
    }
}

private enum DiagnosticConfiguration {
    static let isAvailable = [
        "SAM3_DIAGNOSTIC_BUNDLE",
        "SAM3_DIAGNOSTIC_MANIFEST",
        "SAM3_DIAGNOSTIC_OUTPUT",
    ].allSatisfy { ProcessInfo.processInfo.environment[$0] != nil }
}

private struct PhotoCase: Decodable {
    let id: String
    let image: String
    let concepts: [String]
}

private struct Report: Encodable {
    let modelIdentity: String
    let modelAsset: String
    let operatingSystem: String
    let inputMaxSide: Int
    let boxCoordinateSystem: String
    let expectedRunCount: Int
    let complete: Bool
    let runs: [Run]
}

private struct Run: Encodable {
    let photoID: String
    let imagePath: String
    let imageWidth: Int
    let imageHeight: Int
    let concept: String
    let maximumSegmentCount: Int
    let repetition: Int
    let milliseconds: Double
    let peakResidentBytes: Int64
    let probabilityMapPresent: Bool
    let segmentCount: Int
    let segments: [SegmentRecord]
    let overlay: String?
}

private struct SegmentRecord: Encodable {
    let score: Float
    let boxX: Double
    let boxY: Double
    let boxWidth: Double
    let boxHeight: Double
    let maskWidth: Int
    let maskHeight: Int
    let foregroundPixels: Int
    let maskFingerprint: String
    let maskFile: String?

    init(_ segment: Segment, maskFile: String?) {
        score = segment.score
        boxX = Double(segment.box.origin.x)
        boxY = Double(segment.box.origin.y)
        boxWidth = Double(segment.box.width)
        boxHeight = Double(segment.box.height)
        maskWidth = segment.maskWidth
        maskHeight = segment.maskHeight
        foregroundPixels = segment.mask.reduce(0) { $0 + ($1 ? 1 : 0) }
        var hash: UInt64 = 14_695_981_039_346_656_037
        for pixel in segment.mask {
            hash = (hash ^ (pixel ? 1 : 0)) &* 1_099_511_628_211
        }
        maskFingerprint = String(hash, radix: 16)
        self.maskFile = maskFile
    }
}

private enum DiagnosticError: Error {
    case cannotDecode(String)
    case cannotWrite(String)
    case invalidMask
}
