import CoreAIImageSegmenter
@testable import CoreAISAM3Backend
import CoreGraphics
import Foundation
import ImageIO
import PhotoAIContracts
import PhotoAIStorage
import PhotoAIWorkflows
import Testing

struct ObjectSegmentationTests {
    @Test func conceptValidation() throws {
        let concept = try SegmentationConcept("  Musk   Ox  ")
        #expect(concept.query == "Musk Ox")
        #expect(concept.cacheIdentifier == "musk ox")
        #expect(try SegmentationConcept("musk ox").cacheIdentifier == concept.cacheIdentifier)
        #expect(throws: SegmentationConceptError.empty) { try SegmentationConcept("   ") }
        #expect(throws: SegmentationConceptError.controlCharacter) { try SegmentationConcept("fox\ncar") }
        #expect(throws: SegmentationConceptError.tooLong) {
            try SegmentationConcept(String(repeating: "x", count: 257))
        }
        #expect(SegmentationConcept(.birdHead).query == "bird head")
    }

    @Test func separateMasksSortAndBoxes() throws {
        let first = segment([true, false, false, false], score: 0.4,
                            box: CGRect(x: 0, y: 0, width: 1, height: 1))
        let second = segment([false, false, false, true], score: 0.9,
                             box: CGRect(x: -1, y: 0, width: 3, height: 1))
        let empty = segment([false, false, false, false], score: 1, box: .zero)
        let instances = try CoreAISAM3Provider.decodeInstances(
            [first, second, empty], inputSize: CGSize(width: 4, height: 1),
            outputSize: CGSize(width: 8, height: 2), limit: 8
        )
        #expect(instances.count == 2)
        #expect(instances.map(\.score) == [0.9, 0.4])
        #expect(instances.map(\.id) == ["0", "1"])
        #expect(instances[0].normalizedBoundingBox == CGRect(x: 0.75, y: 0, width: 0.25, height: 1))
        #expect(instances[1].normalizedBoundingBox == CGRect(x: 0, y: 0, width: 0.25, height: 1))
        #expect(instances.allSatisfy { $0.mask.width == 8 && $0.mask.height == 2 })
        let native = try CoreAISAM3Provider.decodeInstances(
            [first, second], inputSize: CGSize(width: 4, height: 1),
            outputSize: CGSize(width: 4, height: 1), limit: 8
        )
        #expect(native[0].mask.dataProvider.flatMap { $0.data as Data? } == Data([0, 0, 0, 255]))
        #expect(native[1].mask.dataProvider.flatMap { $0.data as Data? } == Data([255, 0, 0, 0]))
        #expect(try CoreAISAM3Provider.decodeInstances([], inputSize: CGSize(width: 4, height: 1),
                    outputSize: CGSize(width: 4, height: 1), limit: 8).isEmpty)
        #expect(try CoreAISAM3Provider.decodeInstances([first, second], inputSize: CGSize(width: 4, height: 1),
                    outputSize: CGSize(width: 4, height: 1), limit: 1).count == 1)
        let repeatResult = try CoreAISAM3Provider.decodeInstances(
            [first, second, empty], inputSize: CGSize(width: 4, height: 1),
            outputSize: CGSize(width: 8, height: 2), limit: 8
        )
        #expect(repeatResult.map(\.id) == instances.map(\.id))
    }

    @Test func diskCachePreservesIndividualInstances() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PhotoAIKit-ObjectTest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try ObjectMaskDiskStore(cacheDirectory: directory)
        let source = AIImageSource(id: UUID(), url: directory.appendingPathComponent("photo.jpg"),
                                   displayName: "photo")
        let concept = try SegmentationConcept("musk ox")
        let model = ModelIdentity(family: "sam3", name: "test", assetName: "test.aimodel")
        let key = ObjectMaskStorageKey(source: source,
                                       sourceIdentity: SourceFileIdentity(fileSize: 42, modificationDate: nil),
                                       concept: concept, modelIdentity: model,
                                       inputMaxSide: 4320, maximumInstanceCount: 8)
        let masks = try CoreAISAM3Provider.decodeInstances(
            [segment([true, false, false, false], score: 0.9, box: .zero),
             segment([false, false, false, true], score: 0.8, box: .zero)],
            inputSize: CGSize(width: 4, height: 1),
            outputSize: CGSize(width: 4, height: 1), limit: 8
        )
        let result = ObjectSegmentationResult(
            sourceID: source.id, requestID: UUID(), concept: concept, instances: masks,
            modelIdentity: model, inputSize: CGSize(width: 4, height: 1),
            outputSize: CGSize(width: 4, height: 1), timing: .init()
        )
        try await store.save(result, for: key)
        let cached = await store.load(for: key)
        #expect(cached?.instances.map(\.id) == ["0", "1"])
        #expect(cached?.instances.map(\.score) == [0.9, 0.8])
        #expect(cached?.instances.map(\.normalizedBoundingBox) == masks.map(\.normalizedBoundingBox))
        #expect(cached?.requestID == result.requestID)
        let changedKey = ObjectMaskStorageKey(source: source,
                                              sourceIdentity: SourceFileIdentity(fileSize: 43, modificationDate: nil),
                                              concept: concept, modelIdentity: model,
                                              inputMaxSide: 4320, maximumInstanceCount: 8)
        #expect(await store.load(for: changedKey) == nil)
    }

    @Test func serviceCachesAndPassesInstanceLimit() async throws {
        let provider = FakeObjectProvider()
        let store = ObjectMaskMemoryStore()
        let service = try ObjectSegmentationService(provider: provider, stores: [store],
                                                    maxSide: 2, maximumInstanceCount: 3)
        let image = try #require(CoreAISAM3Provider.decodeInstances(
            [segment([true, false, false, false], score: 1, box: .zero)],
            inputSize: CGSize(width: 4, height: 1),
            outputSize: CGSize(width: 4, height: 1), limit: 1
        ).first?.mask)
        let source = AIImageSource(id: UUID(), url: URL(fileURLWithPath: "/tmp/absent-photo.jpg"),
                                   displayName: "photo")
        let concept = try SegmentationConcept("musk ox")
        let first = try await service.segment(image: image, source: source, concept: concept)
        let second = try await service.segment(image: image, source: source, concept: concept)
        #expect(first.requestID == second.requestID)
        #expect(first.instances[0].mask.width == 4)
        #expect(await provider.callCount == 1)
        #expect(await provider.lastLimit == 3)
        #expect(await provider.lastWidth == 2)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["SAM3_PHASE1_BUNDLE"] != nil &&
                       ProcessInfo.processInfo.environment["SAM3_PHASE1_IMAGE"] != nil))
    func localModelReturnsSeparateInstances() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundle = try #require(environment["SAM3_PHASE1_BUNDLE"])
        let imagePath = try #require(environment["SAM3_PHASE1_IMAGE"])
        let source = try #require(CGImageSourceCreateWithURL(
            URL(fileURLWithPath: imagePath) as CFURL, nil
        ))
        let image = try #require(CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 4_320,
        ] as CFDictionary))
        let provider = try CoreAISAM3Provider(modelBundleURL: URL(fileURLWithPath: bundle))
        let request = ObjectSegmentationRequest(
            sourceID: UUID(), concept: try SegmentationConcept("musk ox"), image: image,
            inputSize: CGSize(width: image.width, height: image.height),
            outputSize: CGSize(width: image.width, height: image.height),
            maxSide: 4_320, maximumInstanceCount: 8
        )
        let result = try await provider.segmentInstances(request)
        #expect(result.instances.count >= 2)
        #expect(result.instances.count <= 8)
        #expect(result.instances.allSatisfy { $0.mask.width == image.width && $0.mask.height == image.height })
        #expect(result.instances.map(\.id) == result.instances.indices.map(String.init))
        #expect(result.instances.filter { $0.score >= 0.5 }.count >= 2)
    }

    private func segment(_ mask: [Bool], score: Float, box: CGRect) -> Segment {
        Segment(mask: mask, maskWidth: 4, maskHeight: 1, box: box, score: score)
    }
}

private actor FakeObjectProvider: ObjectInstanceSegmenting {
    nonisolated let modelIdentity = ModelIdentity(family: "sam3", name: "fake", assetName: "fake.aimodel")
    private(set) var callCount = 0
    private(set) var lastLimit = 0
    private(set) var lastWidth = 0

    func segmentInstances(_ request: ObjectSegmentationRequest) async throws -> ObjectSegmentationResult {
        callCount += 1
        lastLimit = request.maximumInstanceCount
        lastWidth = request.image.width
        let instance = ObjectMaskInstance(index: 0, mask: request.image, score: 0.9,
                                          normalizedBoundingBox: CGRect(x: 0, y: 0, width: 1, height: 1))
        return ObjectSegmentationResult(
            sourceID: request.sourceID, requestID: request.requestID, concept: request.concept,
            instances: [instance], modelIdentity: modelIdentity,
            inputSize: request.inputSize, outputSize: request.inputSize, timing: .init()
        )
    }
}
