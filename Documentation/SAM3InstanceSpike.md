# SAM 3 instance capability spike

This opt-in diagnostic tests the exact SAM 3 Core AI bundle intended for
RawCull. It does not change PhotoAIKit's public segmentation API or the
exhaustive mask used by Deep Review.

Create a JSON manifest outside the repository. Paths are relative to the
manifest file, or absolute:

```json
[
  {"id":"separated-birds","image":"separated-birds.jpg","concepts":["bird"]},
  {"id":"overlapping-birds","image":"overlapping-birds.jpg","concepts":["bird"]},
  {"id":"near-and-far","image":"near-and-far.jpg","concepts":["bird"]},
  {"id":"no-car","image":"no-car.jpg","concepts":["car"]}
]
```

Include representative `person`, `car`, and `face` cases when available. Run
the diagnostic with a release build on the Mac that will be used for the
decision:

```sh
cd /path/to/PhotoAIKit
SAM3_DIAGNOSTIC_BUNDLE=/path/to/SAM3 \
SAM3_DIAGNOSTIC_MANIFEST=/path/to/cases.json \
SAM3_DIAGNOSTIC_OUTPUT=/path/to/spike-results \
swift test -c release --filter SAM3InstanceCapabilityTests
```

The test creates `sam3-instance-report.json`, individual mask PNGs, and colored
instance overlays. Without all three environment variables, the test is
skipped.
The JSON report is rewritten after each run and includes a `complete` flag,
so partial results survive a stopped test.
It orients and downsizes each image to a maximum side of 4,320 pixels, then
runs each concept twice at segment caps of 5, 8, 12, and 16. The report records
segment counts, scores, pixel-coordinate boxes, nonempty mask pixel counts,
mask fingerprints, semantic-map presence, end-to-end call time, and
the process's high-water resident memory. The first run also includes model
loading; later runs show warm behavior. Peak memory is a process-wide high
water mark, so use a fresh test process for another comparison.

Inspect the overlays against the source photographs. A nonzero segment count
alone does not establish that separate subjects were found: reject duplicated,
merged, empty, or misplaced masks. Compare the two repetitions using mask
fingerprints and boxes to assess ordering stability. Check the no-match cases
for spurious detections. Record the Core AI dependency revision, model
fingerprint, source image set, practical cap, and go/no-go decision in
`Docs/sam3qwen.md`. If the shipped model yields only the semantic probability
map or unusable instance masks, stop the Objects implementation.

No model bundle or test photographs should be checked into source control.
