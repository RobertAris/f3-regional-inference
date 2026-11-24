# Reproducing the results

The [benchmark tables](../evidence/tables.md) give task accuracy, latency, memory, and the effect of changing the requested regions. The [one-page note](../paper/regional-f3.pdf) and [technical report](../paper/technical.pdf) summarize the findings. To run the model on your own events, use the [inference guide](inference.md).

## Check the published numbers

The [benchmark supplement](../evidence/benchmarks.zip) contains the measurement records, original protocols, source snapshots, benchmark scripts and tests. From the root of a fresh clone, unpack it once with Python 3.11 or later. The numerical replays below need NumPy. They run on CPU without the recording datasets or model weights.

```sh
python -m zipfile -e evidence/benchmarks.zip .
python evidence/replay_results.py
python -m followup.precision_policy.replay
python -m followup.precision_policy.sensitivity
python -m followup.region_fidelity.replay
python -m followup.multitask.replay --verify-source
```

| Replay | What it checks |
|---|---|
| Preview ablation | Flow disagreement with ordinary filters, projected filters and native first-response sampling |
| Original timing comparison | Ten regions on each of 43 packets against the fastest dense method meeting both fidelity limits |
| Timing sensitivity | The same regional timings against every finite dense alternative, including FP16 |
| Individual regions | The four corner errors and the effect of supplying the exact normalization factors |
| Perception benchmarks | Feature fidelity, labeled task accuracy, crossover, throughput and memory. Every measured source file is checked against its recorded SHA-256. |

Flow timings use one packet per recording and five interleaved repeats. The feature and global-head panels use separate development recordings and three fixed packets per evaluation recording. Those timers include region selection. The flow timers start after the regions have been chosen. All GPU measurements come from one Apple M5 Max.

## Ground truth and new measurements

The [image-row controls](../evidence/spatial-controls.json) include per-packet depth correlations and semantic confusion matrices. Image row predicts inverse-depth rank. Each semantic row predicts its most frequent class across the three development masks for that domain. To rescore the same labeled regions, restore the MVSEC and M3ED depth packets and both DSEC semantic panels as described in `followup/multitask/REPRODUCE.md` in the supplement. With NumPy and SciPy installed, run:

```sh
python -m followup.multitask.spatial_controls --data-root datasets
```

The supplement includes the following instructions and entry points.

| Task | Path after extraction |
|---|---|
| Restore perception data and rerun the exact measured model versions | `followup/multitask/REPRODUCE.md` |
| Prepare the four-panel flow benchmark | `followup/REPRODUCE.md` |
| Repeat the flow timings or investigate corner errors | `followup/precision_policy/README.md` and `followup/region_fidelity/README.md` |
| Replay the moving-mask and stream results | `python followup/consumer/replay.py` and `python followup/service/replay.py` |
| Run the 36 regression tests | `python -m unittest discover -s followup/multitask -p 'test_*.py'` |

Install the [pinned benchmark dependencies](../requirements.txt) for fresh model runs and the full test suite. Use the two measured source snapshots for exact reruns. To benchmark changes to the extracted code, commit the benchmark sources in a separate scratch checkout first so the new records have a resolvable Git revision.

## Build the papers

From `paper/`, run each command twice with TeX Live.

```sh
pdflatex -halt-on-error -jobname=regional-f3 regional.tex
pdflatex -halt-on-error technical.tex
```
