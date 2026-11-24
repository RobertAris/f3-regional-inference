# Regional inference for Fast Feature Field

Can an event camera deliver F3 features and optical flow sooner when an application needs only a few locations? On one Apple M5 Max, ten 32 × 32 regions arrived **7.72–8.84% sooner** across four camera panels than the fastest measured dense implementation with finite output. Four of 430 regions exceeded 0.50 px disagreement with dense flow. The largest error was 2.09 px.

[One-page research note](paper/regional-f3.pdf) · [Technical report](paper/technical.pdf)

## How it works

F3 normalizes each channel using responses across the image. Cropping the input changes those responses, even where a crop contains all the neighboring pixels needed for convolution. We run the first five encoder blocks over the full packet, estimate the remaining normalization factors on a coarse grid, and finish only the requested regions with the original weights and native output resolution.

Correcting the preview filters for their wider spacing accounts for most of the improvement. In a controlled bilinear comparison on M3ED, flow disagreement fell from 0.684 px with an ordinary preview to 0.094 px after scale correction. Sampling the first preview response directly on the native feature map reduced it to 0.053 px. On Prophesee the corresponding errors were 1.208, 0.326 and 0.089 px. [Method and ablation](paper/technical.pdf)

## What was faster?

Each flow request returned all 32 features and both flow components to the host. The original comparison required dense candidates to meet the field and flow tolerances. A retrospective check gave dense inference the fastest finite candidate on each packet. Regional inference was faster on all 43 packets in both comparisons. All ten regions met both tolerances on 40 packets. [Timings and per-region errors](docs/reproduction.md)

| Mean fresh-request latency saved | Fastest finite dense | Fastest dense within tolerance |
|---|---:|---:|
| Prophesee EVIMO2v2 | 8.35% | 13.17% |
| Samsung EVIMO2v2 | 8.70% | 10.29% |
| M3ED | 8.84% | 8.84% |
| E-MOTION | 7.72% | 7.72% |

For **features alone**, four event-selected 48 × 48 regions were 4.11% faster on DSEC and 8.36% faster on M3ED than dense FP16. Dense FP16 was faster on MVSEC and closer to the FP32 reference on average in all three panels.

For **depth and segmentation**, exact requests retain global attention. At FP32, regional depth averaged 1.64 ms slower on MVSEC and 0.80 ms faster on M3ED. Regional FP32 segmentation reproduced every requested class. At matched FP16 the optimized dense decoder was faster on all 18 packets. With five outputs sharing the original F3 encoder on one DSEC packet, dense heads took 194.54 ms and regional heads took 200.11 ms. [Task accuracy, latency and memory](evidence/tables.md)

[Run inference](docs/inference.md) · [Reproduce the measurements](docs/reproduction.md) · [Original F3 paper](https://arxiv.org/html/2509.25146v2)
