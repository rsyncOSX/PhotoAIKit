# Extraction map

This map records how the CLIP and SAM3 code reviewed in `RawCullSAM3/isolateai.md` is represented in PhotoAIKit. It distinguishes reusable AI behavior from integration code that intentionally stays in the host application.

## SAM3

| RawCullSAM3 source | Package destination | Result |
|---|---|---|
| `SubjectSegmentationTypes.swift` | `PhotoAIContracts/SubjectSegmentation.swift`, `SubjectMaskStorage.swift` | Public, UI-neutral, typed, `Sendable` request/result/diagnostic/progress contracts and provider/store protocols. Prompt display labels stay in RawCull. |
| `CoreAISAM3Provider.swift` | `CoreAISAM3Backend/CoreAISAM3Provider.swift` | Full tokenizer, lazy load, inference, query selection, sigmoid conversion, bilinear resize, threshold, feather, diagnostics, and flat-bundle shim. Model bundle URL is required. |
| `SAM3ModelIdentity.swift` | `ModelIdentity.swift`, `ModelBundleResolver.swift` | Generic identity plus SAM3-compatible cache identifier. No static installed-model lookup. |
| `SAM3ModelResourceManager.swift` | `ModelBundleResolver.swift` and `.sam3` descriptor | Validation is reusable and URL-driven. RawCull paths, bundle fallback, and display strings are removed. |
| `SubjectSegmentationActor.swift` | `PhotoAIWorkflows/SegmentationService.swift` | Preprocessing, resizing, cache lookup/persistence, provider invocation, partitioning, prefetch, cancellation, and progress use `AIImageSource`. Global cache and `FileItem` dependencies are removed. “Latest request wins” is deliberately left to the host. |
| `SubjectMaskCache.swift` | `PhotoAIStorage/SubjectMaskMemoryStore.swift` | Public injected actor store. |
| `SAM3MaskDiskCache.swift` | `PhotoAIStorage/SubjectMaskDiskStore.swift` | PNG/JSON load, validation, save, size, pruning, clear, modification date, and fast metadata check. Cache directory is required; logging is package-local. |
| `SAM3MaskGenerationPipeline.swift` | `PhotoAIWorkflows/SegmentationBatchPipeline.swift` | Package-owned sources, summary/events, translated cached/generated/failed progress. |
| `SAM3MaskBuildTypes.swift` | `SegmentationBatchPipeline.swift` | Transport-neutral summary and event values extracted. RawCull helper request fields are host-only. |
| `SAM3MaskInventoryEntry.swift` | `SubjectMaskGeometry.swift` | Alpha extraction, coverage, normalized bounds, centroid, and freshness. |
| `SubjectQualityBadgeModel.swift` | `SubjectMaskQuality.swift` | Geometry thresholds, clipped-edge detection, and quality classification. RawCull badge labels/help text remain host presentation. |
| `SAM3SubjectMaskCacheReader.swift` | `PhotoAIWorkflows/SubjectMaskRepository.swift` | Injected cache facade with explicit prompt/model/size configuration. RawCull's static reader is removed. |
| `SAM3MaskCatalogIndex.swift` | Host adapter over `SubjectMaskStoring` + `SubjectMaskGeometry` | The reusable read/geometry operations are extracted; progressive catalog publication remains tied to the host catalog model. |
| `SAM3MaskHelperController.swift` | Not packaged | AppKit process launch, executable lookup, security scope, parent PID, app path, and restart are RawCull-specific per `isolateai.md`. |
| `SAM3MaskHelperProgressView.swift`, segmentation controls/settings/views | Not packaged | SwiftUI and app display state stay in RawCull. |
| SAM3 consumers in sharpness, burst review, comparison, thumbnails, and zoom | Not packaged | These remain app adapters/policy and now receive a narrow mask facade from RawCull's `RawCullAIContainer`. |

## CLIP

| RawCullSAM3 source | Package destination | Result |
|---|---|---|
| `CoreAICLIPProvider.swift` | `CoreAICLIPBackend/CoreAICLIPProvider.swift` | Full lazy Core AI load, descriptor validation, dummy token/mask creation, image resize and CLIP normalization, NDArray fill/flatten, run, and normalized typed output. Model bundle URL is required. |
| `CLIPModelResourceManager.swift` | `ModelBundleResolver.swift` and `.clip` descriptor | URL validation and identity are reusable. RawCull paths, bundle fallback, and display text are removed. |
| `SimilarityEmbeddingEnvelope` normalization/cosine helpers | `ImageEmbedding.swift`, `SimilarityMetric.swift` | Typed normalized vectors and backend/model-compatible cosine distance. |
| Version-1 CLIP JSON persistence | `LegacyCLIPEmbeddingCodec.swift` | Compatible envelope for staged RawCull integration. `EmbeddingCodec` provides the new fully typed envelope. |
| Bounded indexing loop and progress | `EmbeddingIndexer.swift` | Package-owned sources, injected decoder/provider, configurable concurrency, per-item or whole-batch fallback, typed failures/results. |
| RawParserKit and ImageIO thumbnail decoding | Host `ImageDecoding` adapter | RAW format behavior remains owned by the integrating app. |
| Vision feature-print generation/archive/distance | Host fallback provider/codec | Vision is a product-selected alternative backend, not CLIP inference. The package's fallback policy accepts any provider. |
| `SimilarityScoringModel` observable state, settings, estimation/status strings | Not packaged | UI/presentation state stays in RawCull. Settings/model URL selection now occurs in the RawCull integration layer, and burst grouping work is delegated to a separate RawCull policy model. |
| Burst adjacency cache/grouping, saliency mismatch, review/rating decisions | Not packaged | Culling policy consumes embeddings but is not CLIP functionality. |

## Non-runtime sources and assets

- `tools/export_clip.py`, `tools/export_sam3.py`, and `tools/select_sam3_asset.py` are exporter/developer tools, not runtime package code, matching the scope decision in `isolateai.md`.
- `RawCullSAM3/Resources/Models/CLIP` and `RawCullSAM3/Resources/Models/SAM3` are not copied.
- The package manifest declares no resource target and no model files.

## Boundary checks

The package source contains no imports or references to `RawCullCore`, `RawParserKit`, `FileItem`, `SharedMemoryCache`, `SettingsViewModel`, SwiftUI, Observation, AppKit, or `Bundle.main`. Public API tests import the modules normally (never with `@testable`) and use fake host source/decoder/provider implementations. RawCull constructs package providers, stores, workflows, helper paths, and host adapters once in `RawCullAIContainer`.
