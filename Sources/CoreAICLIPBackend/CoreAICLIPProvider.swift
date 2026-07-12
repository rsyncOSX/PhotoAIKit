import CoreAI
import CoreAIImageSegmenter
import CoreGraphics
import Foundation
import PhotoAIContracts

/// Actor-owned Core AI CLIP runtime. The host supplies the model bundle URL.
public actor CoreAICLIPProvider: ImageEmbeddingProviding {
    public nonisolated let modelIdentity: ModelIdentity

    private let modelBundleURL: URL
    private var loadedModel: LoadedCLIPModel?

    public init(modelBundleURL: URL) throws {
        let resolver = ModelBundleResolver(descriptor: .clip)
        guard case let .valid(_, identity) = resolver.status(at: modelBundleURL) else {
            throw CLIPProviderError.invalidModelBundle(resolver.status(at: modelBundleURL))
        }
        self.modelBundleURL = modelBundleURL
        self.modelIdentity = identity
    }

    public func embedding(for image: CGImage) async throws -> ImageEmbedding {
        let model = try await loadModel()
        let values = try await imageEmbedding(for: image, model: model)
        return ImageEmbedding(
            backend: "clip",
            modelIdentity: modelIdentity,
            values: values
        )
    }

    private func imageEmbedding(for image: CGImage, model: LoadedCLIPModel) async throws -> [Float] {
        let imageInput = try Self.makeImageInput(image, descriptor: model.imageDescriptor)
        let tokenInput = Self.makeTokenInput(model.dummyTokens, descriptor: model.inputIDsDescriptor)
        let attentionMaskInput = Self.makeAttentionMaskInput(descriptor: model.attentionMaskDescriptor)

        var outputs = try await model.function.run(inputs: [
            model.imageInputName: imageInput,
            model.inputIDsInputName: tokenInput,
            model.attentionMaskInputName: attentionMaskInput,
        ])
        guard let embeddingOutput = outputs.remove(model.imageEmbedsOutputName)?.ndArray else {
            throw CLIPProviderError.invalidModel("CLIP image embedding output is missing.")
        }
        let values = Self.flattenAsFloat(embeddingOutput)
        guard !values.isEmpty else {
            throw CLIPProviderError.invalidModel("CLIP image embedding output is empty.")
        }
        return values
    }

    private func loadModel() async throws -> LoadedCLIPModel {
        if let loadedModel { return loadedModel }

        let metadataURL = modelBundleURL.appendingPathComponent("metadata.json")
        let metadata = try JSONDecoder().decode(
            ModelBundleMetadata.self,
            from: Data(contentsOf: metadataURL)
        )
        guard let assetName = metadata.assets["main"] else {
            throw CLIPProviderError.invalidModel("metadata.json does not define assets.main.")
        }
        let modelURL = modelBundleURL.appendingPathComponent(assetName)
        let tokenizer = try CLIPTokenizer(
            folder: modelBundleURL.appendingPathComponent("tokenizer", isDirectory: true)
        )

        let model = try await AIModel(contentsOf: modelURL, options: Self.specializationOptions())
        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw CLIPProviderError.invalidModel("Cannot find main function in CLIP model.")
        }
        guard let function = try model.loadFunction(named: "main") else {
            throw CLIPProviderError.invalidModel("Cannot load main function from CLIP model.")
        }

        let imageInputName = try Self.requiredName("pixel_values", kind: "input", names: descriptor.inputNames)
        let inputIDsInputName = try Self.requiredName("input_ids", kind: "input", names: descriptor.inputNames)
        let attentionMaskInputName = try Self.requiredName("attention_mask", kind: "input", names: descriptor.inputNames)
        let imageEmbedsOutputName = try Self.requiredName("image_embeds", kind: "output", names: descriptor.outputNames)

        guard case let .ndArray(imageDescriptor) = descriptor.inputDescriptor(of: imageInputName),
              case let .ndArray(inputIDsDescriptor) = descriptor.inputDescriptor(of: inputIDsInputName),
              case let .ndArray(attentionMaskDescriptor) = descriptor.inputDescriptor(of: attentionMaskInputName)
        else {
            throw CLIPProviderError.invalidModel("CLIP inputs are not NDArrays.")
        }
        guard imageDescriptor.shape.count == 4,
              inputIDsDescriptor.shape.count == 2,
              attentionMaskDescriptor.shape.count == 2
        else {
            throw CLIPProviderError.invalidModel(
                "Unexpected CLIP input shapes: image=\(imageDescriptor.shape), input_ids=\(inputIDsDescriptor.shape), attention_mask=\(attentionMaskDescriptor.shape)."
            )
        }

        let textBatchSize = inputIDsDescriptor.shape[0]
        let sequenceLength = inputIDsDescriptor.shape[1]
        let dummyTokens = Array(
            repeating: tokenizer.encode("a photo", contextLength: sequenceLength),
            count: textBatchSize
        )
        let loaded = LoadedCLIPModel(
            function: function,
            imageInputName: imageInputName,
            inputIDsInputName: inputIDsInputName,
            attentionMaskInputName: attentionMaskInputName,
            imageEmbedsOutputName: imageEmbedsOutputName,
            imageDescriptor: imageDescriptor,
            inputIDsDescriptor: inputIDsDescriptor,
            attentionMaskDescriptor: attentionMaskDescriptor,
            dummyTokens: dummyTokens
        )
        loadedModel = loaded
        return loaded
    }

    private nonisolated static func specializationOptions() -> SpecializationOptions {
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = false
        return options
    }

    private nonisolated static func requiredName(
        _ preferredName: String,
        kind: String,
        names: [String]
    ) throws -> String {
        guard names.contains(preferredName) else {
            throw CLIPProviderError.invalidModel(
                "CLIP \(kind) \(preferredName) is missing. Available names: \(names)."
            )
        }
        return preferredName
    }

    private nonisolated static func makeImageInput(
        _ image: CGImage,
        descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        let shape = descriptor.shape
        let batchSize = shape[0]
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        guard batchSize == 1, channels == 3 else {
            throw CLIPProviderError.invalidModel(
                "Expected CLIP image input shape [1, 3, H, W], got \(shape)."
            )
        }
        let pixels = try preprocessCLIPImage(image, width: width, height: height)
        var array = NDArray(descriptor: descriptor)
        if descriptor.scalarType == .float16 {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                fillNDArray(&array, as: Float16.self, with: pixels.map(Float16.init))
            #else
                throw CLIPProviderError.invalidModel("Float16 CLIP input is not supported on Intel macOS.")
            #endif
        } else {
            fillNDArray(&array, as: Float.self, with: pixels)
        }
        return array
    }

    private nonisolated static func makeTokenInput(
        _ tokens: [[Int32]],
        descriptor: NDArrayDescriptor
    ) -> NDArray {
        let batchSize = descriptor.shape[0]
        let sequenceLength = descriptor.shape[1]
        var array = NDArray(descriptor: descriptor)
        fillNDArray(&array, as: Int32.self, count: batchSize * sequenceLength) { index in
            let row = index / sequenceLength
            let column = index % sequenceLength
            guard row < tokens.count, column < tokens[row].count else {
                return CLIPTokenizer.eotTokenId
            }
            return tokens[row][column]
        }
        return array
    }

    private nonisolated static func makeAttentionMaskInput(
        descriptor: NDArrayDescriptor
    ) -> NDArray {
        let count = descriptor.shape[0] * descriptor.shape[1]
        var array = NDArray(descriptor: descriptor)
        fillNDArray(&array, as: Int32.self, count: count) { _ in 1 }
        return array
    }

    private nonisolated static func preprocessCLIPImage(
        _ image: CGImage,
        width: Int,
        height: Int
    ) throws -> [Float] {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rgba,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              )
        else { throw CLIPProviderError.imagePreprocessingFailed }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let count = width * height
        var chw = [Float](repeating: 0, count: 3 * count)
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let standardDeviation: [Float] = [0.26862954, 0.26130258, 0.27577711]
        for pixel in 0 ..< count {
            let offset = pixel * bytesPerPixel
            let red = Float(rgba[offset]) / 255
            let green = Float(rgba[offset + 1]) / 255
            let blue = Float(rgba[offset + 2]) / 255
            chw[pixel] = (red - mean[0]) / standardDeviation[0]
            chw[count + pixel] = (green - mean[1]) / standardDeviation[1]
            chw[2 * count + pixel] = (blue - mean[2]) / standardDeviation[2]
        }
        return chw
    }

    private nonisolated static func fillNDArray<T: BitwiseCopyable>(
        _ array: inout NDArray,
        as _: T.Type,
        with elements: some Collection<T>
    ) {
        var view = array.mutableView(as: T.self)
        view.copyElements(fromContentsOf: elements)
    }

    private nonisolated static func fillNDArray<T: BitwiseCopyable>(
        _ array: inout NDArray,
        as _: T.Type,
        count: Int,
        using generator: (Int) -> T
    ) {
        var view = array.mutableView(as: T.self)
        view.withUnsafeMutablePointer { pointer, _, _ in
            for index in 0 ..< count { pointer[index] = generator(index) }
        }
    }

    private nonisolated static func flattenAsFloat(_ array: NDArray) -> [Float] {
        switch array.scalarType {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            flattenNDArray(array, as: Float16.self)
        #endif
        case .float32:
            flattenNDArray(array, as: Float.self)
        default:
            []
        }
    }

    private nonisolated static func flattenNDArray<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray,
        as _: T.Type
    ) -> [Float] {
        let total = array.shape.reduce(1, *)
        var result = [Float](repeating: 0, count: total)
        array.view(as: T.self).withUnsafePointer { pointer, _, _ in
            for index in 0 ..< total { result[index] = Float(pointer[index]) }
        }
        return result
    }

    private struct LoadedCLIPModel {
        let function: InferenceFunction
        let imageInputName: String
        let inputIDsInputName: String
        let attentionMaskInputName: String
        let imageEmbedsOutputName: String
        let imageDescriptor: NDArrayDescriptor
        let inputIDsDescriptor: NDArrayDescriptor
        let attentionMaskDescriptor: NDArrayDescriptor
        let dummyTokens: [[Int32]]
    }
}

public enum CLIPProviderError: Error, CustomStringConvertible, Sendable {
    case invalidModelBundle(ModelBundleStatus)
    case invalidModel(String)
    case imagePreprocessingFailed

    public var description: String {
        switch self {
        case let .invalidModelBundle(status): "Invalid CLIP model bundle: \(status)"
        case let .invalidModel(message): message
        case .imagePreprocessingFailed: "CLIP image preprocessing failed."
        }
    }
}

public extension ModelBundleDescriptor {
    static let clip = ModelBundleDescriptor(
        family: "clip",
        fallbackName: "CLIP",
        requiredRelativePaths: ["tokenizer/tokenizer.json"]
    )
}
