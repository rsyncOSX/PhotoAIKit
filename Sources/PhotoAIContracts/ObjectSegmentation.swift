import CoreGraphics
import Foundation

public enum SegmentationConceptError: Error, Equatable, Sendable {
    case empty
    case tooLong
    case controlCharacter
}

/// Open-vocabulary query. The normalized query is also its locale-independent cache key.
public struct SegmentationConcept: Codable, Hashable, Sendable {
    public static let maximumUTF8Bytes = 256
    public let query: String
    public let cacheIdentifier: String

    public init(_ input: String) throws {
        guard !input.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
            throw SegmentationConceptError.controlCharacter
        }
        let normalizedSpace = input.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        guard !normalizedSpace.isEmpty else { throw SegmentationConceptError.empty }
        guard normalizedSpace.utf8.count <= Self.maximumUTF8Bytes else {
            throw SegmentationConceptError.tooLong
        }
        query = normalizedSpace
        cacheIdentifier = normalizedSpace.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .precomposedStringWithCanonicalMapping
    }

    public init(_ prompt: SubjectSegmentationPrompt) {
        // Enum queries are fixed, nonempty, and already below the limit.
        self = try! SegmentationConcept(prompt.query)
    }

    private enum CodingKeys: String, CodingKey { case query }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(container.decode(String.self, forKey: .query))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(query, forKey: .query)
    }
}

public struct ObjectSegmentationRequest: Sendable {
    public let requestID: UUID
    public let sourceID: UUID
    public let concept: SegmentationConcept
    public let image: CGImage
    public let inputSize: CGSize
    public let outputSize: CGSize
    public let maxSide: Int
    public let maximumInstanceCount: Int

    public init(requestID: UUID = UUID(), sourceID: UUID, concept: SegmentationConcept,
                image: CGImage, inputSize: CGSize, outputSize: CGSize,
                maxSide: Int, maximumInstanceCount: Int = 8) {
        self.requestID = requestID
        self.sourceID = sourceID
        self.concept = concept
        self.image = image
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.maxSide = maxSide
        self.maximumInstanceCount = maximumInstanceCount
    }
}

public struct ObjectMaskInstance: Identifiable, Sendable {
    /// Result-local stable rank after deterministic sorting.
    public let id: String
    public let index: Int
    public let mask: CGImage
    public let score: Float
    /// Unit coordinates with the same origin as Core AI's macOS input-image boxes.
    public let normalizedBoundingBox: CGRect

    public init(index: Int, mask: CGImage, score: Float, normalizedBoundingBox: CGRect) {
        self.index = index
        self.id = String(index)
        self.mask = mask
        self.score = score
        self.normalizedBoundingBox = normalizedBoundingBox
    }
}

public struct ObjectSegmentationResult: Sendable {
    public let sourceID: UUID
    public let requestID: UUID
    public let concept: SegmentationConcept
    public let instances: [ObjectMaskInstance]
    public let modelIdentity: ModelIdentity
    public let inputSize: CGSize
    public let outputSize: CGSize
    public let timing: SubjectSegmentationTiming

    public init(sourceID: UUID, requestID: UUID, concept: SegmentationConcept,
                instances: [ObjectMaskInstance], modelIdentity: ModelIdentity,
                inputSize: CGSize, outputSize: CGSize, timing: SubjectSegmentationTiming) {
        self.sourceID = sourceID
        self.requestID = requestID
        self.concept = concept
        self.instances = instances
        self.modelIdentity = modelIdentity
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.timing = timing
    }
}

public protocol ObjectInstanceSegmenting: Sendable {
    var modelIdentity: ModelIdentity { get }
    func segmentInstances(_ request: ObjectSegmentationRequest) async throws -> ObjectSegmentationResult
}

public enum ObjectSegmentationError: Error, Equatable, Sendable {
    case invalidInstanceLimit
    case invalidImageSize
    case decodeFailure
}
