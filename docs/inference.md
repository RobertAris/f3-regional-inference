# Running regional inference

The demo returns native-resolution F3 features and optical flow for selected rectangles in one event packet. Its default is the FP32 bicubic configuration reported in the [main results](../README.md): five full-resolution encoder blocks, a stride-two normalization preview, and a native regional suffix. See [the method](../paper/technical.pdf) for the normalization and filter projection details.

## Model and environment

Use Python 3.11 or later with PyTorch, NumPy, PyYAML, and the dependencies of the [pinned upstream F3 checkout](https://github.com/grasp-lyrl/fast-feature-fields/tree/320eec6debf4491058d19a905b450f8de7fe4722). Check out upstream revision `320eec6debf4491058d19a905b450f8de7fe4722` and obtain these released files:

- The [20 ms M3ED car-day encoder checkpoint](https://richeek-penn.s3.amazonaws.com/prolev/ff/patchff_fullcardaym3ed_small_20ms.pth).
- The matching M3ED flow-head checkpoint and its YAML configuration, following the [upstream release instructions](https://github.com/grasp-lyrl/fast-feature-fields/tree/320eec6debf4491058d19a905b450f8de7fe4722).

The bundled loaders verify the checkpoint and flow-configuration hashes and check that the flow checkpoint contains the same encoder weights. Keep the model files outside this repository.

## Input arrays

Save two NumPy arrays as `.npy` files:

| Array | Shape | Contents |
|---|---|---|
| Events | `[N, 3]` | Integer rows `(relative_microseconds, x, y)`, sorted by time. Timestamps are relative to the packet end and lie in `[-20000, 0)`. |
| Rectangles | `[Q, 4]` | Integer rows `(x, y, width, height)` in native sensor pixels. Use at least one rectangle; all rectangles must have the same positive width and height and lie within the sensor. |

The released encoder does not use event polarity. Sensor dimensions are given as `width height`; the event coordinates and rectangles must use that geometry. The published timing workload uses ten 32 × 32 rectangles, although the interface accepts other numbers and sizes.

## Run the demo

From the repository root:

```sh
python -m query_f3.inference.demo \
  --f3-repo /path/to/fast-feature-fields \
  --encoder-weights /path/to/patchff_fullcardaym3ed_small_20ms.pth \
  --head-weights /path/to/last.pth \
  --head-config /path/to/flow_config.yaml \
  --size 1280 720 --device mps \
  --projection bicubic \
  --events events.npy --rectangles rectangles.npy \
  --output regions.npz
```

`--projection bicubic` is the default and uses the existing `CubicPreview` implementation from the precision experiments. Select `--projection bilinear` to run the earlier bilinear variant. Both use FP32 model parameters and the same native prefix and suffix. No retraining is required.

The default device is `cpu`; the CLI also accepts `mps` and `cuda`. The recorded benchmarks used MPS on an Apple M5 Max.

## Outputs

The output archive contains:

| Key | Shape | Meaning |
|---|---|---|
| `features` | `[Q, 32, width, height]` | The learned F3 feature vectors in each requested region. |
| `flow100ms` | `[Q, 2, width, height]` | Horizontal and vertical displacement, in native pixels per 100 ms. |
| `rectangles` | `[Q, 4]` | The input rectangles, in the original order. |

Spatial axes are **width first**, unlike the usual image-array convention. Overlapping rectangles are returned separately. The 19-pixel computation halo is not included in the output.

For comparison with displacement labels over an interval `dt_us`, scale `flow100ms` by `dt_us / 100000`, assuming constant velocity over the interval. The CLI uses time zero as the packet end. The reported ground-truth-aligned experiments instead describe a 20 ms packet centered on the flow source timestamp, so they include 10 ms of events after that timestamp; preserve that alignment when reproducing those comparisons.

## Reusing the model

The CLI loads weights for each process invocation. For a service, load the encoder and head once and retain a `CubicPreview` instance so its projected kernels can be reused. Recompute the native prefix and normalization factors for each new packet, then call `batched_suffix` with that packet's rectangles. The demo shows this sequence directly.

Reported request timings start with a new packet and resident model weights. They include packet construction, inference, gathering the requested outputs, and transfer to the host; they exclude model loading, dataset I/O, rectification, and region selection. A previously computed dense field can be sliced directly for repeated queries.

See [reproduction instructions](reproduction.md) for replaying saved measurements and obtaining the experimental data. The [individual-region results](../paper/technical.pdf) describe the observed approximation errors.

## Depth, semantic labels, features and future occupancy

The multi-task interface accepts different native rectangles for each consumer, including mixed sizes and duplicate requests. Coordinates use the same integer event array described above, with an optional fourth polarity column. Use sorted events from one 20 ms packet; rectify coordinates before calling the service when working with rectified labels.

Install the [perception environment](../requirements.txt). Place the original [depth weights](https://richeek-penn.s3.amazonaws.com/prolev/monoculardepth/dav2b_fullm3ed_pseudo_518x518x20/best.pth) in `checkpoints/multitask/depth.pth` and the [semantic weights](https://richeek-penn.s3.amazonaws.com/prolev/segmentation/segformer_b3_fullm3ed_800x600x20/models/best_miou.pth) in `checkpoints/multitask/semantics.pth`. Save the [SegFormer B3 architecture](https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024/resolve/7fd6d07f04d2511d4efd22e0fd99c004555433b6/config.json) as `segformer.json` alongside them. The loader checks their SHA-256 and every encoder tensor in each task checkpoint. Only requested heads need to be present.

For example, save this request as `regions.json` for a 640 × 480 camera.

```json
{"features":[[100,80,32,32]],"depth":[[210,120,48,32]],"semantics":[[212,148,16,16]],"occupancy":[[8,8,24,24]]}
```

```sh
python -m query_f3.multitask.infer --method exact --device mps \
  --size 640 480 --events events.npy --requests regions.json --output answers.npz \
  --upstream ../fast-feature-fields --encoder checkpoints/f3/patchff_fullcardaym3ed_small_20ms.pth \
  --models checkpoints/multitask
```

Each `task_i` array has shape `[channels,height,width]`; the companion `task_rectangles` array preserves request order. Depth returns relative inverse depth. Semantic output contains eleven class logits. Occupancy returns twenty future-bin logits. Features have 32 channels. Add `"flow"` and pass `--flow-weights` and `--flow-config` to receive horizontal and vertical native displacement in pixels per 100 ms. `--method dense` runs the reference full-frame heads. `--method exact` shares the native field, retains global attention and samples the requested decoder outputs. `--method preview --split 3` uses a shared projected normalization preview for feature and occupancy requests. Use `cpu` on systems without MPS. The CLI processes every supplied event; the benchmark's deterministic high-density cap is part of its preprocessing.
