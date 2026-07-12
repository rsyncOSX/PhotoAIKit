# PhotoAIKit

Reusable Core AI CLIP and SAM3 code extracted from RawCullSAM3. The package owns model inference, typed contracts, reusable indexing/segmentation workflows, and optional storage. It does not contain RawCull view models, UI, RAW-file decoding, helper-process behavior, or culling policy.

## Requirements

- Xcode 27 or newer (`swift-tools-version: 6.4`)
- macOS 27 or newer
- Swift 6 language mode

The package pins the same `apple/coreai-models` revision used by the source application. It does not embed any `.aimodel`, `.aimodelc`, tokenizer, or model metadata resources.

## Products

- `PhotoAIContracts`: model identities, source values, segmentation/embedding types, provider/store/decoder protocols, and URL-based model-bundle validation.
- `CoreAICLIPBackend`: actor-owned CLIP preprocessing and Core AI inference.
- `CoreAISAM3Backend`: actor-owned SAM3 tokenization, inference, and mask decoding.
- `PhotoAIWorkflows`: bounded embedding indexing, configurable fallback, cosine similarity, segmentation preprocessing/batching, mask geometry, and quality classification.
- `PhotoAIStorage`: injected memory/disk mask stores and current/legacy embedding codecs.

## Model URLs

The host application locates, downloads, bookmarks, or otherwise manages models. It then passes each bundle URL directly to the backend:

```swift
import CoreAICLIPBackend
import CoreAISAM3Backend

let clip = try CoreAICLIPProvider(modelBundleURL: clipBundleURL)
let sam3 = try CoreAISAM3Provider(modelBundleURL: sam3BundleURL)
```

A bundle URL is expected to contain:

```text
ModelBundle/
├── metadata.json              # assets.main names the selected model asset
├── tokenizer/
│   └── tokenizer.json
└── selected-model.aimodel     # or .aimodelc
```

No default Application Support path, `Bundle.main` lookup, or package resource fallback is used.

## Host integration

Map the app's photo type to `AIImageSource`, provide an `ImageDecoding` adapter, and inject model/cache URLs:

```swift
import PhotoAIContracts
import PhotoAIStorage
import PhotoAIWorkflows

let source = AIImageSource(
    id: photo.id,
    url: photo.url,
    displayName: photo.name
)

let memory = SubjectMaskMemoryStore()
let disk = try SubjectMaskDiskStore(cacheDirectory: appMaskCacheURL)
let service = SegmentationService(
    provider: sam3,
    stores: [memory, disk],
    maxSide: 4_320
)

let result = try await service.segment(
    image: decodedImage,
    source: source,
    prompt: .subject
)
```

For CLIP indexing, inject the host decoder and choose fallback explicitly:

```swift
let indexer = EmbeddingIndexer(
    primaryProvider: clip,
    fallbackProvider: optionalHostFallback,
    decoder: rawAwareDecoder,
    fallbackPolicy: .wholeBatch,
    concurrencyLimit: 2
)

let index = try await indexer.index(sources)
```

`LegacyCLIPEmbeddingCodec` reads and writes the version-1 JSON envelope currently used by RawCull, which allows later integration without an immediate embedding-data migration.

## Deliberate host responsibilities

- model download/install UI and model URL selection;
- security-scoped bookmarks and sandbox access;
- RAW decoding (for example RawParserKit) and ImageIO fallback;
- Vision feature-print implementation, if retained;
- SwiftUI/Observation state and display text;
- “latest selection wins” behavior;
- burst grouping, saliency penalties, sharpness/rating decisions;
- SAM3 helper-process launch, cancellation, parent-process coordination, and app restart.

See [Documentation/ExtractionMap.md](Documentation/ExtractionMap.md) for the complete source-to-package audit.
