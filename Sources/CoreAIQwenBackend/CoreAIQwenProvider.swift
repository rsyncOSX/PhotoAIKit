import CoreAILanguageModels
import Foundation
import PhotoAIContracts

/// Metadata needed by a host to describe and budget a Qwen model.
public struct QwenModelConfiguration: Equatable, Sendable {
    public let name: String
    public let tokenizerIdentifier: String
    public let vocabularySize: Int
    public let maximumContextLength: Int
    public let hasEmbeddedTokenizer: Bool
    public let compression: String?
    public let assetName: String
}

/// Validates a Qwen Core AI bundle and creates Foundation Models-compatible runtimes.
///
/// The provider is immutable and `Sendable`. Each returned `CoreAILanguageModel`
/// owns its own lazy resource lifecycle; retain and reuse that model for multiple
/// `LanguageModelSession` values instead of constructing one per request.
public struct CoreAIQwenProvider: Sendable {
    public let modelIdentity: ModelIdentity
    public let configuration: QwenModelConfiguration

    public static let resourceDescriptor = ModelResourceDescriptor.qwen

    public static var factory: ModelProviderFactory<CoreAIQwenProvider> {
        ModelProviderFactory(descriptor: resourceDescriptor) { url in
            try CoreAIQwenProvider(modelBundleURL: url)
        }
    }

    private let modelBundleURL: URL

    public init(modelBundleURL: URL) throws {
        let resolver = ModelBundleResolver(descriptor: Self.resourceDescriptor.bundleDescriptor)
        let status = resolver.status(at: modelBundleURL)
        guard case let .valid(_, identity) = status else {
            throw QwenProviderError.invalidModelBundle(status)
        }

        let metadataURL = modelBundleURL.appendingPathComponent("metadata.json")
        let metadata: Metadata
        do {
            metadata = try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: metadataURL))
        } catch {
            throw QwenProviderError.invalidMetadata(Self.message(for: error))
        }

        guard metadata.kind == "llm" else {
            throw QwenProviderError.unsupportedModelKind(metadata.kind)
        }
        let identifiers = [metadata.language.tokenizer, metadata.source?.huggingFaceModelID]
            .compactMap { $0?.lowercased() }
        guard identifiers.contains(where: { $0.contains("qwen") }) else {
            throw QwenProviderError.notQwenModel
        }
        guard metadata.language.vocabularySize > 0,
              metadata.language.maximumContextLength > 0
        else {
            throw QwenProviderError.invalidMetadata(
                "language.vocab_size and language.max_context_length must be positive."
            )
        }

        self.modelBundleURL = modelBundleURL
        self.modelIdentity = identity
        self.configuration = QwenModelConfiguration(
            name: metadata.name,
            tokenizerIdentifier: metadata.language.tokenizer,
            vocabularySize: metadata.language.vocabularySize,
            maximumContextLength: metadata.language.maximumContextLength,
            hasEmbeddedTokenizer: metadata.language.hasEmbeddedTokenizer,
            compression: metadata.compression,
            assetName: identity.assetName
        )
    }

    /// Creates a Core AI language model compatible with `LanguageModelSession`.
    /// The default lazy mode loads the tokenizer now and defers the model engine
    /// until the first response or an explicit `load()` call.
    public func makeLanguageModel(
        mode: CoreAILanguageModel.LoadMode = .lazy,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto
    ) async throws -> CoreAILanguageModel {
        do {
            return try await CoreAILanguageModel(
                resourcesAt: modelBundleURL,
                mode: mode,
                variant: variant,
                kvCacheStrategy: kvCacheStrategy
            )
        } catch {
            throw QwenProviderError.modelLoad(Self.message(for: error))
        }
    }

    private static func message(for error: Error) -> String {
        let description = error.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        return description.isEmpty ? String(reflecting: error) : description
    }
}

public enum QwenProviderError: Error, CustomStringConvertible, Sendable {
    case invalidModelBundle(ModelBundleStatus)
    case invalidMetadata(String)
    case unsupportedModelKind(String)
    case notQwenModel
    case modelLoad(String)

    public var description: String {
        switch self {
        case let .invalidModelBundle(status):
            "Invalid Qwen model bundle: \(status)"
        case let .invalidMetadata(message):
            "Invalid Qwen metadata: \(message)"
        case let .unsupportedModelKind(kind):
            "Expected an llm bundle, got \(kind)."
        case .notQwenModel:
            "The bundle tokenizer and source model do not identify a Qwen model."
        case let .modelLoad(message):
            "Qwen model load failed: \(message)"
        }
    }
}

public extension ModelBundleDescriptor {
    static let qwen = ModelBundleDescriptor(
        family: ModelResourceDescriptor.qwen.bundleDescriptor.family,
        fallbackName: ModelResourceDescriptor.qwen.bundleDescriptor.fallbackName,
        assetKey: ModelResourceDescriptor.qwen.bundleDescriptor.assetKey,
        requiredRelativePaths: ModelResourceDescriptor.qwen.bundleDescriptor.requiredRelativePaths,
        acceptedAssetExtensions: ModelResourceDescriptor.qwen.bundleDescriptor.acceptedAssetExtensions
    )
}

private extension CoreAIQwenProvider {
    struct Metadata: Decodable {
        let kind: String
        let name: String
        let language: Language
        let source: Source?
        let compression: String?

        struct Language: Decodable {
            let tokenizer: String
            let vocabularySize: Int
            let maximumContextLength: Int
            let hasEmbeddedTokenizer: Bool

            enum CodingKeys: String, CodingKey {
                case tokenizer
                case vocabularySize = "vocab_size"
                case maximumContextLength = "max_context_length"
                case hasEmbeddedTokenizer = "embedded_tokenizer"
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                tokenizer = try container.decode(String.self, forKey: .tokenizer)
                vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
                maximumContextLength = try container.decode(Int.self, forKey: .maximumContextLength)
                hasEmbeddedTokenizer = try container.decodeIfPresent(
                    Bool.self,
                    forKey: .hasEmbeddedTokenizer
                ) ?? true
            }
        }

        struct Source: Decodable {
            let huggingFaceModelID: String?

            enum CodingKeys: String, CodingKey {
                case huggingFaceModelID = "hf_model_id"
            }
        }
    }
}
