import Foundation

/// Stable identity for a model bundle supplied by a host application.
public struct ModelIdentity: Codable, Hashable, Sendable {
    public let family: String
    public let name: String
    public let sourceModel: String?
    public let assetName: String
    public let metadataVersion: String?
    private let cacheIdentifierOverride: String?

    public init(
        family: String,
        name: String,
        sourceModel: String? = nil,
        assetName: String,
        metadataVersion: String? = nil,
        cacheIdentifier: String? = nil
    ) {
        self.family = family
        self.name = name
        self.sourceModel = sourceModel
        self.assetName = assetName
        self.metadataVersion = metadataVersion
        self.cacheIdentifierOverride = cacheIdentifier
    }

    public var cacheIdentifier: String {
        if let cacheIdentifierOverride {
            return cacheIdentifierOverride
        }
        if family.lowercased() == "sam3" {
            return "coreai-sam3-local:\(name):\(assetName)"
        }
        return [family, name, sourceModel ?? "", assetName, metadataVersion ?? ""]
            .joined(separator: ":")
    }
}

public struct ModelBundleMetadata: Codable, Equatable, Sendable {
    public let name: String?
    public let family: String?
    public let sourceModel: String?
    public let metadataVersion: String?
    public let assets: [String: String]

    public init(
        name: String? = nil,
        family: String? = nil,
        sourceModel: String? = nil,
        metadataVersion: String? = nil,
        assets: [String: String]
    ) {
        self.name = name
        self.family = family
        self.sourceModel = sourceModel
        self.metadataVersion = metadataVersion
        self.assets = assets
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case family
        case sourceModel = "source_model"
        case metadataVersion = "metadata_version"
        case assets
    }
}
