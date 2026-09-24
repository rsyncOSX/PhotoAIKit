import Foundation

public struct ObjectMaskStorageKey: Codable, Hashable, Sendable {
    public let source: AIImageSource
    public let sourceIdentity: SourceFileIdentity
    public let concept: SegmentationConcept
    public let modelIdentity: ModelIdentity
    public let inputMaxSide: Int
    public let maximumInstanceCount: Int

    public init(source: AIImageSource, sourceIdentity: SourceFileIdentity,
                concept: SegmentationConcept, modelIdentity: ModelIdentity,
                inputMaxSide: Int, maximumInstanceCount: Int) {
        self.source = source
        self.sourceIdentity = sourceIdentity
        self.concept = concept
        self.modelIdentity = modelIdentity
        self.inputMaxSide = inputMaxSide
        self.maximumInstanceCount = maximumInstanceCount
    }
}

public protocol ObjectMaskStoring: Sendable {
    func load(for key: ObjectMaskStorageKey) async -> ObjectSegmentationResult?
    func save(_ result: ObjectSegmentationResult, for key: ObjectMaskStorageKey) async throws
}
