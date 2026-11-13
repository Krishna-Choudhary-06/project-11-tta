# Project 11 - Test-Time Adaptation

Project files are organized into baseline training and evaluation (`phase1_baseline`),
test-time adaptation (`phase2_tent`), datasets, models, utilities, and experiment results.

Downloaded datasets, model checkpoints, local editor settings, and generated
per-batch results are excluded from Git. The compact Phase 3 aggregate table
(`experiments/phase3_results/master_results.csv`) is kept in the repository.

## Set up the environment

Use Python 3.11 or newer. From the project root, create and activate a virtual
environment, then install the project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch and TorchVision builds matching your
CUDA setup using the official PyTorch installation selector, then install the
remaining packages from `requirements.txt`.

The CIFAR-10 training data downloads automatically when running training.
CIFAR-10-C is a separate manual download; follow the instructions below. The
trained checkpoint is generated locally by training and is not included in Git.

## Train and evaluate the CIFAR-10 baseline

From the project root, run:

```powershell
python -m phase1_baseline.train
python -m phase1_baseline.evaluate
```

Training downloads CIFAR-10 if it is not already present, trains a ResNet-18
for 15 epochs by default, and saves its weights to
`models/checkpoints/resnet18_cifar10.pth`. Evaluation reports test loss,
accuracy, per-class accuracy, and the confusion matrix.

## Test-time adaptation (TENT)

Check the entropy calculation with:

```powershell
python -m phase2_tent.entropy
```

After training the baseline and placing the extracted CIFAR-10-C `.npy` files
(`labels.npy` and corruption arrays) in `data/cifar10_c`, compare a corruption:

```powershell
python -m phase2_tent.evaluate_tent --corruption gaussian_noise --severity 1
```

Use any corruption array name present in `data/cifar10_c` and a severity from
1 to 5. The comparison evaluates the clean source model, corrupted source
model, and a fresh TENT model loaded from the same checkpoint. TENT scores
each corrupted batch before adapting on its images; labels are used only for
evaluation metrics.

The notebook `experiments/phase1_phase2_experiments.ipynb` documents E1–E7,
including clean/corrupted source evaluation, entropy, TENT on Gaussian noise,
blur and fog, and a learning-rate comparison. It writes one row per evaluated
batch to CSV files under `experiments/baseline_results` and
`experiments/tent_results`, including batch accuracy, adaptation entropy,
predicted counts per class, marginal entropy, and learning rate.

### Download CIFAR-10-C

The evaluator does not automatically download CIFAR-10-C. Download the
official archive from Zenodo (about 2.92 GB) and extract it under
`data/cifar10_c`:

```powershell
New-Item -ItemType Directory -Force .\data\cifar10_c | Out-Null
curl.exe -L --fail --retry 5 -o .\data\cifar10_c\CIFAR-10-C.tar "https://zenodo.org/api/records/2535967/files/CIFAR-10-C.tar/content"
Get-FileHash .\data\cifar10_c\CIFAR-10-C.tar -Algorithm MD5
tar -xf .\data\cifar10_c\CIFAR-10-C.tar -C .\data\cifar10_c
```

The expected MD5 checksum is
`56bf5dcef84df0e2308c6dcbcbbd8499`. After extraction, the `data\cifar10_c` or its `CIFAR-10-C` subfolder should
contain files such as `gaussian_noise.npy` and `labels.npy`. This data is
ignored by Git and must be downloaded separately on each machine.

## Phase 3 distribution-shift streams

Inspect the predefined static-shift experiment (Gaussian noise repeated for
three 10,000-image segments):

```powershell
python -m datasets.stream --experiment static --severity 3
```

Inspect the changing-shift experiment (Gaussian noise, then motion blur, then
fog):

```powershell
python -m datasets.stream --experiment changing --severity 3
```

Both use severity 3 and preserve stream order. The static experiment repeats
the same CIFAR-10-C severity-3 images in each segment; the changing experiment
uses the corresponding severity-3 block from each corruption. Batches do not
cross segment boundaries, so each distribution transition is explicit.

Run continual TENT with one persistent model and optimizer across the full
stream:

```powershell
python -m phase3.continual --experiment static --severity 3
python -m phase3.continual --experiment changing --severity 3
```

Each command loads the source checkpoint once, adapts using unlabeled images,
and saves pre-update batch metrics to `experiments/phase3_results`. For a
short smoke test, add `--max-batches 2`; omit it to process the entire stream.
The CSV includes prediction counts, accuracy, conditional entropy, marginal
entropy, normalized marginal entropy (`marginal_entropy / log(10)`), and the
dominant-class prediction ratio. Marginal entropy and dominant-class ratio
help monitor whether predictions are concentrating on fewer classes.
It also records exact per-class mean softmax probabilities and five-batch
rolling marginal entropy and dominant-class ratio. The rolling window is
sample-weighted, so a short batch at a corruption boundary contributes only
its actual number of images.

Time-to-collapse is defined here as both normalized marginal entropy below
`0.20` and dominant-class ratio at least `0.90` for five consecutive batches.
After a stream run, the console summary and adjacent `.summary.json` report the
first batch of that five-batch interval, images processed at its start, the
confirmation batch, and images processed by confirmation. If the criteria are
never met, those values are `null` and collapse was not observed.

Generate the three time-series plots from a Phase 3 CSV:

```powershell
python -m utils.visualization .\experiments\phase3_results\changing_severity_3_lr0.00025.csv
```

The plots (marginal entropy, accuracy, and predicted class distribution) are
saved beside the CSV. The entropy plot overlays batch-wise and five-batch
rolling marginal entropy. Collapse detection is reported separately for
batch-wise metrics and rolling metrics, with the rolling detector evaluated
only after a complete five-batch window is available. Existing Phase 3 CSVs
without rolling-metric columns must be regenerated by rerunning
`python -m phase3.continual` before plotting.

### Phase 3 controlled stress tests

Run one stress test at a time; each run starts from the original checkpoint
with a new model and optimizer. Use `--max-batches 2` to smoke-test a command
before running a complete stream:

```powershell
python -m phase3.stress_tests --stress repeated-noise
python -m phase3.stress_tests --stress learning-rate-sweep
python -m phase3.stress_tests --stress class-imbalance
python -m phase3.stress_tests --stress biased-initialization
python -m phase3.stress_tests --stress batch-size-sweep
```

`repeated-gaussian` is also accepted as an alias for `repeated-noise`.
The repeated-noise test uses ten consecutive Gaussian-noise segments. The
learning-rate sweep compares `0.00025`, `0.0005`, `0.001`, and `0.002`.
The class-imbalance test samples a 30,000-image stream with class 0 at 70%
frequency and the remaining classes balanced; labels are used only to build
the benchmark stream and are never passed to the adaptation function. The
biased-initialization test overwrites the source classifier bias to favor
class 0 before TENT is configured. The batch-size sweep compares 8, 16, 32,
and 64. Each result gets its own CSV and summary JSON under
`experiments/phase3_results`. After a run, `master_results.csv` is updated
with accuracy, the minimum complete five-batch rolling normalized entropy,
the maximum rolling dominant-class ratio, and the rolling collapse result for
all available completed experiments. `--stress all` runs the entire suite.
