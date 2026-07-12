import Foundation

public struct SubjectMaskStorageKey: Codable, Hashable, Sendable {
    public let source: AIImageSource
    public let sourceIdentity: SourceFileIdentity
    public let prompt: SubjectSegmentationPrompt
    public let modelIdentity: ModelIdentity
    public let inputMaxSide: Int

    public init(
        source: AIImageSource,
        sourceIdentity: SourceFileIdentity,
        prompt: SubjectSegmentationPrompt,
        modelIdentity: ModelIdentity,
        inputMaxSide: Int
    ) {
        self.source = source
        self.sourceIdentity = sourceIdentity
        self.prompt = prompt
        self.modelIdentity = modelIdentity
        self.inputMaxSide = inputMaxSide
    }
}

public protocol SubjectMaskStoring: Sendable {
    func load(for key: SubjectMaskStorageKey) async -> SubjectSegmentationResult?
    func contains(_ key: SubjectMaskStorageKey) async -> Bool
    func save(_ result: SubjectSegmentationResult, for key: SubjectMaskStorageKey) async throws
}

public struct SubjectMaskPrefetchProgress: Equatable, Sendable {
    public let completed: Int
    public let total: Int
    public let cached: Int
    public let generated: Int
    public let failed: Int
    public let currentSourceID: UUID?

    public init(
        completed: Int,
        total: Int,
        cached: Int,
        generated: Int,
        failed: Int,
        currentSourceID: UUID?
    ) {
        self.completed = completed
        self.total = total
        self.cached = cached
        self.generated = generated
        self.failed = failed
        self.currentSourceID = currentSourceID
    }

    public var remaining: Int { max(0, total - completed) }
}
