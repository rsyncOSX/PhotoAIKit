import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import PhotoAIContracts
import UniformTypeIdentifiers

/// Separate object-set cache. Every instance retains its own PNG and metadata.
public actor ObjectMaskDiskStore: ObjectMaskStoring {
    public nonisolated let cacheDirectory: URL
    private static let version = "v1-object-instances"

    public init(cacheDirectory: URL) throws {
        self.cacheDirectory = cacheDirectory
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    }

    public func load(for key: ObjectMaskStorageKey) -> ObjectSegmentationResult? {
        guard !Task.isCancelled else { return nil }
        let directory = directory(for: key)
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
              manifest.version == Self.version, manifest.key == key else { return nil }
        var instances: [ObjectMaskInstance] = []
        for (index, entry) in manifest.instances.enumerated() {
            guard let source = CGImageSourceCreateWithURL(
                directory.appendingPathComponent("\(index).png") as CFURL, nil
            ), let mask = CGImageSourceCreateImageAtIndex(source, 0, nil),
                  mask.width == manifest.outputWidth,
                  mask.height == manifest.outputHeight else { return nil }
            instances.append(ObjectMaskInstance(index: index, mask: mask, score: entry.score,
                                                normalizedBoundingBox: entry.box))
        }
        return ObjectSegmentationResult(
            sourceID: key.source.id, requestID: manifest.requestID,
            concept: key.concept, instances: instances, modelIdentity: key.modelIdentity,
            inputSize: CGSize(width: manifest.inputWidth, height: manifest.inputHeight),
            outputSize: CGSize(width: manifest.outputWidth, height: manifest.outputHeight),
            timing: SubjectSegmentationTiming(totalMilliseconds: 0)
        )
    }

    public func save(_ result: ObjectSegmentationResult, for key: ObjectMaskStorageKey) throws {
        try Task.checkCancellation()
        let directory = directory(for: key)
        let staging = cacheDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        do {
            for (index, instance) in result.instances.enumerated() {
                try Task.checkCancellation()
                let png = NSMutableData()
                guard let destination = CGImageDestinationCreateWithData(
                    png, UTType.png.identifier as CFString, 1, nil
                ) else { throw ObjectSegmentationError.decodeFailure }
                CGImageDestinationAddImage(destination, instance.mask, nil)
                guard CGImageDestinationFinalize(destination) else {
                    throw ObjectSegmentationError.decodeFailure
                }
                try (png as Data).write(to: staging.appendingPathComponent("\(index).png"))
            }
            let manifest = Manifest(version: Self.version, key: key, requestID: result.requestID,
                                    inputWidth: Int(result.inputSize.width),
                                    inputHeight: Int(result.inputSize.height),
                                    outputWidth: Int(result.outputSize.width),
                                    outputHeight: Int(result.outputSize.height),
                                    instances: result.instances.map {
                                        Entry(score: $0.score, box: $0.normalizedBoundingBox)
                                    })
            try JSONEncoder().encode(manifest).write(
                to: staging.appendingPathComponent("manifest.json")
            )
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
            try FileManager.default.moveItem(at: staging, to: directory)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
    }

    private nonisolated func directory(for key: ObjectMaskStorageKey) -> URL {
        let raw = [Self.version, key.source.url.standardized.path,
                   key.concept.cacheIdentifier, key.modelIdentity.artifactIdentifier,
                   String(key.inputMaxSide), String(key.maximumInstanceCount)].joined(separator: "\u{1F}")
        let digest = SHA256.hash(data: Data(raw.utf8))
        let name = digest.map { String(format: "%02x", $0) }.joined()
        return cacheDirectory.appendingPathComponent(name, isDirectory: true)
    }

    private struct Manifest: Codable {
        let version: String
        let key: ObjectMaskStorageKey
        let requestID: UUID
        let inputWidth: Int
        let inputHeight: Int
        let outputWidth: Int
        let outputHeight: Int
        let instances: [Entry]
    }

    private struct Entry: Codable {
        let score: Float
        let box: CGRect
    }
}
