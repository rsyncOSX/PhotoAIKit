import PhotoAIContracts

public actor ObjectMaskMemoryStore: ObjectMaskStoring {
    private var entries: [ObjectMaskStorageKey: ObjectSegmentationResult] = [:]

    public init() {}

    public func load(for key: ObjectMaskStorageKey) -> ObjectSegmentationResult? {
        entries[key]
    }

    public func save(_ result: ObjectSegmentationResult, for key: ObjectMaskStorageKey) {
        entries[key] = result
    }

    public func removeAll() { entries.removeAll() }
}
