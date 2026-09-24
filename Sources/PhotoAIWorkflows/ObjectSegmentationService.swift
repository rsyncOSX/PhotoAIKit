import CoreGraphics
import Foundation
import PhotoAIContracts

/// Independent object-set workflow and cache namespace.
public actor ObjectSegmentationService {
    private let provider: any ObjectInstanceSegmenting
    private let stores: [any ObjectMaskStoring]
    private let maxSide: Int
    private let maximumInstanceCount: Int

    public init(provider: any ObjectInstanceSegmenting,
                stores: [any ObjectMaskStoring] = [],
                maxSide: Int = 4_320, maximumInstanceCount: Int = 8) throws {
        guard maxSide > 0, maximumInstanceCount > 0 else {
            throw ObjectSegmentationError.invalidInstanceLimit
        }
        self.provider = provider
        self.stores = stores
        self.maxSide = maxSide
        self.maximumInstanceCount = maximumInstanceCount
    }

    public func segment(image: CGImage, source: AIImageSource,
                        concept: SegmentationConcept) async throws -> ObjectSegmentationResult {
        try Task.checkCancellation()
        let identity = await Task { @concurrent in
            SourceFileIdentity.read(from: source.url)
        }.value
        let key = ObjectMaskStorageKey(source: source, sourceIdentity: identity,
                                       concept: concept, modelIdentity: provider.modelIdentity,
                                       inputMaxSide: maxSide,
                                       maximumInstanceCount: maximumInstanceCount)
        for store in stores {
            try Task.checkCancellation()
            if let cached = await store.load(for: key) { return cached }
        }
        let bounded = try Self.boundedImage(image, maxSide: maxSide)
        let request = ObjectSegmentationRequest(
            sourceID: source.id, concept: concept, image: bounded,
            inputSize: CGSize(width: bounded.width, height: bounded.height),
            outputSize: CGSize(width: image.width, height: image.height),
            maxSide: maxSide, maximumInstanceCount: maximumInstanceCount
        )
        let result = try await provider.segmentInstances(request)
        try Task.checkCancellation()
        var displayInstances: [ObjectMaskInstance] = []
        for instance in result.instances {
            try Task.checkCancellation()
            let mask = try Self.resizedMask(instance.mask, width: image.width, height: image.height)
            displayInstances.append(ObjectMaskInstance(index: instance.index, mask: mask,
                                                       score: instance.score,
                                                       normalizedBoundingBox: instance.normalizedBoundingBox))
        }
        let displayResult = ObjectSegmentationResult(
            sourceID: result.sourceID, requestID: result.requestID,
            concept: result.concept, instances: displayInstances,
            modelIdentity: result.modelIdentity, inputSize: result.inputSize,
            outputSize: CGSize(width: image.width, height: image.height), timing: result.timing
        )
        for store in stores {
            try Task.checkCancellation()
            try await store.save(displayResult, for: key)
        }
        return displayResult
    }

    private nonisolated static func boundedImage(_ image: CGImage, maxSide: Int) throws -> CGImage {
        let longest = max(image.width, image.height)
        guard longest > maxSide else { return image }
        let scale = CGFloat(maxSide) / CGFloat(longest)
        let width = max(1, Int((CGFloat(image.width) * scale).rounded()))
        let height = max(1, Int((CGFloat(image.height) * scale).rounded()))
        guard let context = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ObjectSegmentationError.decodeFailure
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let bounded = context.makeImage() else { throw ObjectSegmentationError.decodeFailure }
        return bounded
    }

    private nonisolated static func resizedMask(_ mask: CGImage, width: Int,
                                                 height: Int) throws -> CGImage {
        guard mask.width != width || mask.height != height else { return mask }
        guard let context = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            throw ObjectSegmentationError.decodeFailure
        }
        context.interpolationQuality = .none
        context.draw(mask, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resized = context.makeImage() else { throw ObjectSegmentationError.decodeFailure }
        return resized
    }
}
