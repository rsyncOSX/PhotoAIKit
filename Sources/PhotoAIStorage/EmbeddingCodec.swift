import Foundation
import PhotoAIContracts

/// Typed persistence for embeddings. Host applications choose where the data is stored.
public enum EmbeddingCodec {
    public static func encode(_ embedding: ImageEmbedding) throws -> Data {
        try JSONEncoder().encode(Envelope(version: 1, embedding: embedding))
    }

    public static func decode(_ data: Data) throws -> ImageEmbedding {
        try JSONDecoder().decode(Envelope.self, from: data).embedding
    }

    private struct Envelope: Codable {
        let version: Int
        let embedding: ImageEmbedding
    }
}
