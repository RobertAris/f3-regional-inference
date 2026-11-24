# Measurements by task

Latency is the mean of recording means of per-packet median times, in milliseconds. Each packet uses three interleaved timing repeats on the M5 Max GPU. The finite column counts packets with finite outputs. Quality averages use the available finite observations. An em dash indicates an undefined or unmeasured quantity.

## Features on MVSEC

5 recordings, 15 packets.

| Method | ms | Finite | Relative RMS (%) | Worst relative RMS (%) |
| --- | --- | --- | --- | --- |
| Upstream dense FP32 | 13.28 | 15/15 | 0.00002 | 0.00008 |
| Native dense FP32 | 12.43 | 15/15 | 0.00002 | 0.00009 |
| Dense FP16 | 10.25 | 15/15 | 1.10791 | 2.28751 |
| Native GRNs at split 3 | — | 15/15 | 0.00003 | 0.00016 |
| Projected preview, split 1 | 8.70 | 15/15 | 6.75839 | 11.67150 |
| Projected preview, split 3 | 10.88 | 15/15 | 1.30589 | 2.59398 |
| Projected preview, split 5 | 12.09 | 15/15 | 0.14439 | 0.30785 |
| Uncorrected preview, split 3 | 10.08 | 15/15 | 2.09394 | 4.20233 |
| Ordinary coarse | 4.48 | 15/15 | 56.84204 | 75.34336 |
| Cropped encoder | 9.93 | 15/15 | 46.86109 | 70.82307 |

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Dense FP16, is -6.1% (recording-group bootstrap 95% interval -7.3% to -5.6%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Features on DSEC

8 recordings, 24 packets.

| Method | ms | Finite | Relative RMS (%) | Worst relative RMS (%) |
| --- | --- | --- | --- | --- |
| Upstream dense FP32 | 60.27 | 24/24 | 0.00003 | 0.00007 |
| Native dense FP32 | 59.76 | 24/24 | 0.00003 | 0.00004 |
| Dense FP16 | 49.77 | 24/24 | 0.56408 | 0.93735 |
| Native GRNs at split 3 | — | 24/24 | 0.00003 | 0.00007 |
| Projected preview, split 1 | 35.80 | 24/24 | 12.47018 | 31.47396 |
| Projected preview, split 3 | 47.73 | 24/24 | 1.00361 | 2.18284 |
| Projected preview, split 5 | 56.06 | 24/24 | 0.06200 | 0.12871 |
| Uncorrected preview, split 3 | 44.97 | 24/24 | 2.63300 | 5.60238 |
| Ordinary coarse | 26.28 | 24/24 | 79.96627 | 123.75638 |
| Cropped encoder | 17.41 | 24/24 | 57.45144 | 85.51347 |

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Dense FP16, is 4.1% (recording-group bootstrap 95% interval 3.4% to 4.9%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Features on M3ED

6 recordings, 18 packets.

| Method | ms | Finite | Relative RMS (%) | Worst relative RMS (%) |
| --- | --- | --- | --- | --- |
| Upstream dense FP32 | 166.38 | 18/18 | 0.00001 | 0.00011 |
| Native dense FP32 | 183.07 | 18/18 | 0.00001 | 0.00004 |
| Dense FP16 | 150.19 | 18/18 | 0.65108 | 1.73239 |
| Native GRNs at split 3 | — | 18/18 | 0.00001 | 0.00007 |
| Projected preview, split 1 | 96.16 | 18/18 | 12.97553 | 34.60907 |
| Projected preview, split 3 | 137.63 | 18/18 | 1.04813 | 1.55612 |
| Projected preview, split 5 | 168.97 | 18/18 | 0.03384 | 0.05690 |
| Uncorrected preview, split 3 | 128.42 | 18/18 | 3.24195 | 5.24915 |
| Ordinary coarse | 69.54 | 18/18 | 85.96381 | 155.44008 |
| Cropped encoder | 23.42 | 18/18 | 61.11730 | 99.27229 |

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Dense FP16, is 8.4% (recording-group bootstrap 95% interval 7.7% to 9.5%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Stereo on MVSEC

5 recordings, 15 packets.

| Method | ms | Finite | Disparity MAE (px) | Within 1 px (%) | Returned / valid | Returned-only MAE | Same as dense (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native dense FP32 | 26.43 | 15/15 | 10.324 | 22.47 | 592 / 592 | 10.324 | 100.00 |
| Dense FP16 | 22.74 | 15/15 | 10.330 | 21.96 | 592 / 592 | 10.330 | 97.32 |
| Projected preview, split 1 | 32.85 | 15/15 | 10.245 | 22.47 | 592 / 592 | 10.245 | 82.33 |
| Projected preview, split 3 | 28.38 | 15/15 | 10.308 | 22.64 | 592 / 592 | 10.308 | 98.32 |
| Projected preview, split 5 | 27.19 | 15/15 | 10.335 | 22.47 | 592 / 592 | 10.335 | 99.44 |
| Uncorrected preview, split 3 | 26.03 | 15/15 | 10.239 | 22.64 | 592 / 592 | 10.239 | 95.86 |
| Ordinary coarse | 11.70 | 15/15 | 9.516 | 21.11 | 592 / 592 | 9.516 | 11.86 |
| Cropped encoder | 719.76 | 15/15 | 12.017 | 13.68 | 592 / 592 | 12.017 | 19.24 |
| Event surface patch matcher | 4.24 | 15/15 | 11.299 | 17.74 | 592 / 592 | 11.299 | 18.79 |
| Event-count SGBM | 5.07 | 15/15 | 10.077 | 34.12 | 428 / 592 | 10.180 | 5.82 |
| Zero disparity | 0.16 | 15/15 | 10.394 | 8.11 | 592 / 592 | 10.394 | 9.62 |

7 of 592 labeled queries exceed the 48 px search range. MAE includes these queries. The SGBM search extends through 63 px. Abstentions receive zero disparity for the common-support MAE. Agreement covers every selected site, including sites without ground truth.

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Dense FP16, is -24.8% (recording-group bootstrap 95% interval -29.0% to -11.8%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Stereo on DSEC

8 recordings, 24 packets.

| Method | ms | Finite | Disparity MAE (px) | Within 1 px (%) | Returned / valid | Returned-only MAE | Same as dense (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native dense FP32 | 117.70 | 24/24 | 7.696 | 32.84 | 335 / 335 | 7.696 | 100.00 |
| Dense FP16 | 96.78 | 24/24 | 7.636 | 33.43 | 335 / 335 | 7.636 | 98.14 |
| Projected preview, split 1 | 83.01 | 24/24 | 7.852 | 34.33 | 335 / 335 | 7.852 | 74.30 |
| Projected preview, split 3 | 98.54 | 24/24 | 7.679 | 32.84 | 335 / 335 | 7.679 | 99.07 |
| Projected preview, split 5 | 113.51 | 24/24 | 7.696 | 32.84 | 335 / 335 | 7.696 | 99.93 |
| Uncorrected preview, split 3 | 93.05 | 24/24 | 7.882 | 32.54 | 335 / 335 | 7.882 | 96.48 |
| Ordinary coarse | 48.13 | 24/24 | 9.544 | 23.88 | 335 / 335 | 9.544 | 14.61 |
| Cropped encoder | 934.18 | 24/24 | 9.032 | 30.75 | 335 / 335 | 9.032 | 30.21 |
| Event surface patch matcher | 47.55 | 24/24 | 7.559 | 44.18 | 335 / 335 | 7.559 | 19.65 |
| Event-count SGBM | 20.90 | 24/24 | 6.184 | 55.82 | 289 / 335 | 3.044 | 4.18 |
| Zero disparity | 1.62 | 24/24 | 22.845 | 0.00 | 335 / 335 | 22.845 | 5.18 |

21 of 335 labeled queries exceed the 48 px search range. MAE includes these queries. The SGBM search extends through 63 px. Abstentions receive zero disparity for the common-support MAE. Agreement covers every selected site, including sites without ground truth.

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Dense FP16, is -1.8% (recording-group bootstrap 95% interval -4.3% to 1.0%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Depth on MVSEC

5 recordings, 15 packets.

| Method | ms | Finite | Spearman ρ | Defined ρ | Normalized MAE | Relative RMS (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Native dense FP32 | 95.44 | 15/15 | 0.4955 | 15 | 0.9285 | 0.00004 |
| Dense SDPA | 128.62 | 15/15 | 0.4956 | 15 | 0.9285 | 0.00004 |
| FP16 encoder, FP32 head | 93.04 | 15/15 | 0.4938 | 15 | 0.9309 | 0.38161 |
| Dense FP16 | 63.06 | 0/15 | — | 0 | — | — |
| Full context, individual crops | 108.17 | 15/15 | 0.4955 | 15 | 0.9285 | 0.00005 |
| Full context, batched native tail FP32 | 97.08 | 15/15 | 0.4956 | 15 | 0.9285 | 0.00004 |
| Batched native tail FP16 | 64.96 | 0/15 | — | 0 | — | — |
| Coarse field + native regions | 95.77 | 15/15 | 0.4593 | 15 | 0.9385 | 10.83375 |
| Ordinary coarse | 31.80 | 15/15 | 0.6079 | 15 | 0.7991 | 17.54993 |
| Cropped packet and head | 315.40 | 15/15 | 0.1517 | 15 | 1.3294 | 56.49246 |

Whole-frame dense mean ρ 0.6897. Regional accuracy uses the union of the event-selected boxes.

Full context, batched native tail FP32 time saved against the fastest dense configuration finite on every packet, FP16 encoder, FP32 head, is -4.3% (recording-group bootstrap 95% interval -4.4% to -4.1%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Depth on M3ED

6 recordings, 18 packets.

| Method | ms | Finite | Spearman ρ | Defined ρ | Normalized MAE | Relative RMS (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Native dense FP32 | 284.86 | 18/18 | 0.5757 | 18 | 0.9917 | 0.00004 |
| Dense SDPA | 346.59 | 18/18 | 0.5757 | 18 | 0.9917 | 0.00003 |
| FP16 encoder, FP32 head | 252.82 | 18/18 | 0.5752 | 18 | 0.9925 | 0.23617 |
| Dense FP16 | 216.40 | 0/18 | — | 0 | — | — |
| Full context, individual crops | 290.64 | 18/18 | 0.5757 | 18 | 0.9917 | 0.00009 |
| Full context, batched native tail FP32 | 284.07 | 18/18 | 0.5757 | 18 | 0.9917 | 0.00003 |
| Batched native tail FP16 | 218.50 | 0/18 | — | 0 | — | — |
| Coarse field + native regions | 239.87 | 18/18 | 0.6102 | 18 | 0.9454 | 14.09160 |
| Ordinary coarse | 186.75 | 18/18 | 0.5516 | 18 | 0.9679 | 20.55962 |
| Cropped packet and head | 341.61 | 18/18 | -0.1573 | 18 | 1.6524 | 99.59742 |

Whole-frame dense mean ρ 0.7469. Regional accuracy uses the union of the event-selected boxes.

Full context, batched native tail FP32 time saved against the fastest dense configuration finite on every packet, FP16 encoder, FP32 head, is -12.4% (recording-group bootstrap 95% interval -12.7% to -11.9%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Semantics on DSEC manual night

3 recordings, 9 packets.

| Method | ms | Finite | mIoU (%) | Pixel accuracy (%) | Dense label agreement (%) | Relative RMS (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Native dense FP32 | 83.13 | 9/9 | 23.913 | 54.369 | 100.000 | 0.00004 |
| Dense SDPA | 84.87 | 9/9 | 23.913 | 54.369 | 100.000 | 0.00004 |
| Folded dense FP32 | 77.05 | 9/9 | 23.913 | 54.369 | 100.000 | 0.00005 |
| FP16 encoder, FP32 head | 73.81 | 9/9 | 23.886 | 54.307 | 99.802 | 0.27443 |
| Dense FP16 | 54.65 | 9/9 | 23.894 | 54.328 | 99.807 | 0.27215 |
| Folded dense FP16 | 52.52 | 9/9 | 23.895 | 54.329 | 99.803 | 0.27285 |
| Full context, individual crops | 93.76 | 9/9 | 23.913 | 54.369 | 100.000 | 0.00004 |
| Full context, batched native tail FP32 | 79.00 | 9/9 | 23.913 | 54.369 | 100.000 | 0.00005 |
| Batched native tail FP16 | 55.95 | 9/9 | 23.896 | 54.330 | 99.806 | 0.27266 |
| Coarse field + native regions | 66.91 | 9/9 | 19.065 | 45.236 | 71.001 | 29.89849 |
| Ordinary coarse | 63.29 | 9/9 | 24.547 | 49.807 | 67.271 | 45.83468 |
| Cropped packet and head | 51.93 | 9/9 | 7.939 | 27.719 | 42.609 | 64.16063 |

Whole-frame dense mIoU 22.23%. Constant road on the same queries 2.31%. The class mean excludes classes absent from both truth and prediction.

Full context, batched native tail FP32 time saved against the fastest dense configuration finite on every packet, Folded dense FP16, is -50.4% (one route group). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Semantics on DSEC daytime pseudo-labels

3 recordings, 9 packets.

| Method | ms | Finite | mIoU (%) | Pixel accuracy (%) | Dense label agreement (%) | Relative RMS (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Native dense FP32 | 82.65 | 9/9 | 33.066 | 60.298 | 100.000 | 0.00004 |
| Dense SDPA | 84.51 | 9/9 | 33.066 | 60.298 | 100.000 | 0.00004 |
| Folded dense FP32 | 76.36 | 9/9 | 33.066 | 60.298 | 100.000 | 0.00004 |
| FP16 encoder, FP32 head | 73.61 | 9/9 | 33.053 | 60.258 | 99.866 | 0.23354 |
| Dense FP16 | 54.02 | 9/9 | 33.057 | 60.264 | 99.872 | 0.24782 |
| Folded dense FP16 | 52.54 | 9/9 | 33.056 | 60.263 | 99.870 | 0.24759 |
| Full context, individual crops | 93.48 | 9/9 | 33.066 | 60.298 | 100.000 | 0.00003 |
| Full context, batched native tail FP32 | 77.92 | 9/9 | 33.066 | 60.298 | 100.000 | 0.00004 |
| Batched native tail FP16 | 55.67 | 9/9 | 33.059 | 60.264 | 99.870 | 0.24733 |
| Coarse field + native regions | 66.72 | 9/9 | 32.045 | 60.326 | 86.774 | 20.01759 |
| Ordinary coarse | 63.08 | 9/9 | 22.726 | 51.751 | 66.792 | 36.32130 |
| Cropped packet and head | 48.29 | 9/9 | 14.315 | 32.940 | 47.585 | 56.99957 |

Whole-frame dense mIoU 42.53%. Constant road on the same queries 1.38%. The class mean excludes classes absent from both truth and prediction.

Full context, batched native tail FP32 time saved against the fastest dense configuration finite on every packet, Folded dense FP16, is -48.3% (recording-group bootstrap 95% interval -49.2% to -46.6%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Occupancy on ESOT low resolution

4 recordings, 4 packets.

| Method | ms | Finite | Log loss (bits) | Brier score | Dense logit RMSE |
| --- | --- | --- | --- | --- | --- |
| Upstream dense FP32 | 14.45 | 4/4 | 0.79343 | 0.18425 | 0.00000 |
| Native dense FP32 | 13.18 | 4/4 | 0.79343 | 0.18425 | 0.00000 |
| Dense FP16 | 10.91 | 4/4 | 0.79273 | 0.18412 | 0.00786 |
| Projected preview, split 1 | 9.60 | 4/4 | 0.79636 | 0.18486 | 0.04970 |
| Projected preview, split 3 | 11.76 | 4/4 | 0.78937 | 0.18295 | 0.00859 |
| Projected preview, split 5 | 13.15 | 4/4 | 0.79377 | 0.18435 | 0.00096 |
| Ordinary coarse | 5.05 | 4/4 | 0.89712 | 0.21663 | 0.34146 |
| Cropped encoder | 10.74 | 4/4 | 0.77939 | 0.17839 | 0.28415 |
| Past event frequency | 0.47 | 4/4 | 0.10660 | 0.01391 | 3.16136 |

Occupied bins comprise 1.35%. The largest difference between upstream whole-image and gathered 1 × 1 logits is 2.38e-06.

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Dense FP16, is -7.8% (recording-group bootstrap 95% interval -8.3% to -7.3%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Occupancy on ESOT high resolution

4 recordings, 4 packets.

| Method | ms | Finite | Log loss (bits) | Brier score | Dense logit RMSE |
| --- | --- | --- | --- | --- | --- |
| Upstream dense FP32 | 137.29 | 4/4 | 0.83979 | 0.19645 | 0.00000 |
| Native dense FP32 | 159.02 | 4/4 | 0.83979 | 0.19645 | 0.00000 |
| Dense FP16 | 125.82 | 1/4 | 0.67761 | 0.14417 | 0.00581 |
| Projected preview, split 1 | 72.97 | 4/4 | 0.82640 | 0.19215 | 0.03579 |
| Projected preview, split 3 | 113.56 | 4/4 | 0.83483 | 0.19485 | 0.00906 |
| Projected preview, split 5 | 144.88 | 4/4 | 0.83991 | 0.19649 | 0.00031 |
| Ordinary coarse | 46.82 | 4/4 | 0.89515 | 0.21405 | 0.26820 |
| Cropped encoder | 16.18 | 4/4 | 0.75842 | 0.17023 | 0.31067 |
| Past event frequency | 3.02 | 4/4 | 0.09062 | 0.01260 | 3.28530 |

Occupied bins comprise 1.71%. The largest difference between upstream whole-image and gathered 1 × 1 logits is 5.25e-06.

Projected preview, split 3 time saved against the fastest dense configuration finite on every packet, Upstream dense FP32, is 17.3% (recording-group bootstrap 95% interval 17.0% to 17.5%). Negative values mean longer latency. This descriptive comparator imposes no downstream-accuracy threshold.

## Sampled crossover

Numbers are regional / native dense FP32 latency. Values below 1 favor regional execution. Each row holds packet, query extent and placement fixed while count increases. Nonmonotone rows retain every observed value. Each geometry uses one packet per resolution.

| Feature domain | Square side | Layout | 1 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| dsec | 8 | clustered | 0.772 | 0.780 | 0.788 | 0.806 |
| dsec | 8 | dispersed | 0.780 | 0.781 | 0.783 | 0.805 |
| dsec | 32 | clustered | 0.774 | 0.781 | 0.807 | 0.925 |
| dsec | 32 | dispersed | 0.778 | 0.783 | 0.804 | 0.923 |
| dsec | 96 | clustered | 0.781 | 0.826 | 0.987 | 1.695 |
| dsec | 96 | dispersed | 0.789 | 0.814 | 0.994 | 1.691 |
| m3ed | 8 | clustered | 0.722 | 0.725 | 0.719 | 0.722 |
| m3ed | 8 | dispersed | 0.713 | 0.767 | 0.714 | 0.726 |
| m3ed | 32 | clustered | 0.722 | 0.728 | 0.727 | 0.769 |
| m3ed | 32 | dispersed | 0.722 | 0.717 | 0.727 | 0.767 |
| m3ed | 96 | clustered | 0.720 | 0.729 | 0.793 | 1.041 |
| m3ed | 96 | dispersed | 0.722 | 0.736 | 0.788 | 1.047 |
| mvsec | 8 | clustered | 0.852 | 0.866 | 0.854 | 0.948 |
| mvsec | 8 | dispersed | 0.828 | 0.852 | 0.863 | 0.951 |
| mvsec | 32 | clustered | 0.844 | 0.885 | 0.945 | 1.487 |
| mvsec | 32 | dispersed | 0.854 | 0.867 | 0.990 | 1.465 |
| mvsec | 96 | clustered | 0.876 | 1.051 | 1.784 | 4.826 |
| mvsec | 96 | dispersed | 0.869 | 1.029 | 1.800 | 4.794 |

| Head / domain | Square side | Layout | 1 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| depth/m3ed | 8 | clustered | 0.999 | 0.998 | 1.001 | 1.005 |
| depth/m3ed | 8 | dispersed | 1.005 | 1.003 | 1.002 | 1.003 |
| depth/m3ed | 32 | clustered | 1.003 | 1.008 | 1.004 | 1.003 |
| depth/m3ed | 32 | dispersed | 1.009 | 0.998 | 1.004 | 1.011 |
| depth/m3ed | 96 | clustered | 0.999 | 1.006 | 1.008 | 1.033 |
| depth/m3ed | 96 | dispersed | 1.000 | 1.000 | 1.012 | 1.034 |
| depth/mvsec | 8 | clustered | 1.016 | 1.009 | 1.009 | 1.021 |
| depth/mvsec | 8 | dispersed | 1.006 | 0.990 | 1.010 | 1.021 |
| depth/mvsec | 32 | clustered | 1.009 | 1.010 | 1.024 | 1.075 |
| depth/mvsec | 32 | dispersed | 1.010 | 1.016 | 1.028 | 1.067 |
| depth/mvsec | 96 | clustered | 1.014 | 1.033 | 1.137 | 1.482 |
| depth/mvsec | 96 | dispersed | 1.010 | 1.030 | 1.115 | 1.495 |
| semantics/dsec | 8 | clustered | 1.020 | 1.023 | 1.045 | 1.041 |
| semantics/dsec | 8 | dispersed | 1.008 | 1.019 | 1.024 | 1.049 |
| semantics/dsec | 32 | clustered | 1.014 | 1.023 | 1.041 | 1.087 |
| semantics/dsec | 32 | dispersed | 1.023 | 1.028 | 1.044 | 1.100 |
| semantics/dsec | 96 | clustered | 1.027 | 1.050 | 1.105 | 1.345 |
| semantics/dsec | 96 | dispersed | 1.020 | 1.043 | 1.115 | 1.348 |

## Fixed-packet density and lookup

The density experiment uniformly subsamples events before proposal selection. The location can therefore change with density. The table reports fresh-request medians.

| DSEC fraction retained | Dense FP32 ms | Dense FP16 ms | Split 3 ms | Split 3 relative RMS (%) |
| --- | --- | --- | --- | --- |
| 0.1 | 46.22 | 35.31 | 33.73 | 2.411 |
| 0.5 | 51.19 | 41.16 | 39.85 | 1.576 |
| 1 | 59.44 | 49.32 | 47.52 | 0.895 |

| Resident lookup | Table MiB | Setup ms | Dense FP32 ms | Dense FP16 ms | Split 3 ms |
| --- | --- | --- | --- | --- | --- |
| dsec | 187.5 | 331.06 | 47.53 | 36.58 | 34.79 |
| m3ed | 562.5 | 928.42 | 146.25 | 116.16 | 102.19 |
| mvsec | 54.9 | 127.23 | 11.85 | 9.47 | 10.58 |

## Fresh requests, cached requests and memory

Each row uses an isolated process and resident weights. Throughput is a timed sequence of serial fresh requests on the same input contents with the packet context rebuilt each time. Cached requests reuse that packet's context. Reported peaks are sampled at 2 ms and include loaded weights. RSS and Metal driver allocation overlap in unified memory. CPU rows have no Metal allocation. Both standalone MPS flow rows time the same dense flow head. Compare joint-dense with joint-batched for the cost of regional decoding on a shared field.

| Configuration | Fresh ms | Cached ms | Packets/s | Peak RSS MiB | Peak driver MiB | Peak active tensor MiB |
| --- | --- | --- | --- | --- | --- | --- |
| cpu-depth-batched | 855.03 | 6.90 | 1.17 | 2348.0 | 0.0 | 0.0 |
| cpu-depth-dense | 899.92 | 0.01 | 1.12 | 3147.1 | 0.0 | 0.0 |
| cpu-features-dense | 100.50 | 0.10 | 9.91 | 523.4 | 0.0 | 0.0 |
| cpu-features-regional | 80.12 | 8.96 | 12.49 | 529.8 | 0.0 | 0.0 |
| cpu-semantics-batched | 546.81 | 3.22 | 1.85 | 1049.5 | 0.0 | 0.0 |
| cpu-semantics-dense | 563.39 | 0.02 | 1.77 | 1049.1 | 0.0 | 0.0 |
| joint-batched | 200.11 | 8.50 | 4.99 | 1169.7 | 2232.0 | 1068.2 |
| joint-dense | 194.54 | 1.02 | 5.15 | 1150.2 | 2228.0 | 1093.9 |
| joint-separate-batched | 428.28 | 8.86 | 2.33 | 1169.4 | 2235.0 | 1087.2 |
| mps-depth-batched | 97.61 | 3.23 | 10.24 | 873.6 | 1118.0 | 828.5 |
| mps-depth-dense | 95.75 | 0.29 | 10.43 | 866.9 | 1117.8 | 828.5 |
| mps-features-dense | 12.84 | 0.26 | 77.08 | 467.6 | 1051.9 | 82.2 |
| mps-features-regional | 11.66 | 1.75 | 85.87 | 498.0 | 1053.0 | 90.6 |
| mps-flow-dense | 20.36 | 0.35 | 49.03 | 489.2 | 1058.8 | 73.9 |
| mps-flow-regional | 19.94 | 0.20 | 50.19 | 491.1 | 1058.8 | 73.9 |
| mps-occupancy-dense | 13.07 | 0.20 | 76.26 | 467.8 | 1051.4 | 126.3 |
| mps-occupancy-regional | 11.57 | 1.80 | 86.02 | 498.8 | 1051.9 | 90.6 |
| mps-semantics-batched | 26.82 | 5.39 | 35.83 | 812.8 | 1130.2 | 257.3 |
| mps-semantics-dense | 23.83 | 0.25 | 37.97 | 799.4 | 1127.8 | 302.1 |
| stereo-dense | 115.40 | — | 8.65 | 508.4 | 1053.3 | 265.4 |
| stereo-half | 93.28 | — | 10.68 | 514.8 | 1057.3 | 273.6 |
| stereo-raw | 45.51 | — | 21.94 | 335.2 | 0.4 | 0.0 |
| stereo-raw-sgbm | 8.90 | — | 111.87 | 313.2 | 0.4 | 0.0 |
| stereo-regional-3-2 | 99.40 | — | 9.98 | 547.5 | 1053.4 | 265.4 |
