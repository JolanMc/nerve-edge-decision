# Nerve Edge Decision: design note (v0.1)

This note explains what Nerve Edge Decision checks, why each check is there, how it is
evaluated and where the evaluation stops. Every number quoted here comes from
[`results/summary.md`](results/summary.md), which `make all` regenerates.

## 1. The question

Neural networks now run on board Earth-observation satellites; Φ-Sat-1 was the
first demonstrator (Giuffrida et al., 2022). Such a network is validated once,
on the ground. In flight, two things can quietly invalidate that validation:

1. **its inputs drift away from the validation data**: a degraded sensor, an
   unusual scene, a scene type it never saw;
2. **its parameters change**: a single-event upset flips a bit in the memory
   that holds the weights.

Neither shows up in the network's output: a network fed a saturated image, or
running with a corrupted weight, still returns a confident class. Nerve Edge Decision
asks, at every inference, *is this still the validated network, running on the
kind of input it was validated on?*, and it measures what asking costs.

It is a small runtime monitor in the spirit of the "safety cage" architecture
described in ECSS-E-HB-40-02A, the ECSS machine learning handbook (2024). It is
not a qualification argument and claims no compliance with that handbook.

## 2. System under test

| | |
|---|---|
| Task | PlanesNet: 32,000 RGB chips of 20 × 20 pixels cut from PlanetScope imagery (3 m per pixel, California), "plane" vs "no plane" |
| Why this dataset | Real Earth-observation imagery, public (CC BY-SA 4.0), small enough to run every experiment on a laptop CPU |
| Split | By PlanetScope scene: 70 / 10 / 20 % of the chips for training, validation and test, no scene shared between splits, so test chips come from acquisitions never seen in training |
| Network | EdgeNet: five 3 × 3 convolutions (32-32-64-64-128 channels), two max-pools, global average pooling, one linear layer; 140,002 parameters, 11.4 million multiply-accumulates per chip |
| Training | 30 epochs, AdamW, one-cycle schedule; geometric augmentation only (rotations by 90°, flips), so that photometric sensor faults stay outside the training distribution; the epoch with the best validation accuracy is kept |
| Deployment | ONNX, then static INT8 quantisation with ONNX Runtime: QDQ format, per-channel symmetric INT8 weights, asymmetric UINT8 activations, MinMax calibration on 2,000 training chips. The INT8 graph is the artefact under test |

Two checks run at every pipeline execution and stop it if they fail: the
monitored graphs must return logits bit-for-bit identical to the plain graph's,
and the ONNX FP32 graph must match PyTorch (largest absolute difference of the
logits in the reported run: 7.6 × 10⁻⁶).

## 3. What can go wrong (the hazards modelled)

### 3.1 Inputs

Eight sensor faults and acquisition shifts, each at a mild and a strong
severity, are applied to the test chips: Gaussian noise, dead or hot pixels,
column stripes (an uncalibrated push-broom detector), saturation (excess gain),
low light, defocus blur, misregistration of the red band, and a cloud. A far
out-of-distribution set adds 2,000 CIFAR-10 photographs (Krizhevsky, 2009)
resized to 20 × 20, which are not Earth observation at all.

A shift matters only if it hurts the classifier. A condition is called
**harmful** when it costs the INT8 classifier at least 3 accuracy points. The
headline metric is the mean AUROC over the harmful conditions; the mean over all
conditions is reported too, since flagging a harmless shift is closer to a false
alarm than to a detection.

### 3.2 Parameters

The memory under test holds the quantised parameters as the INT8 kernels read
them: 139,360 bytes of INT8 weights and 1,288 bytes of INT32 biases, 1,125,184
bits in total. Each trial flips bits, rebuilds the INT8 model and runs it on
1,000 test chips plus 32 golden inputs.

* **Uniform campaign**: 1, 4, 16, 64 or 256 distinct bits drawn uniformly over
  the whole memory, as single-event upsets would be; several flips stand for
  upsets that accumulate between two scrubs.
* **Targeted campaigns**: the bits that matter most, which a uniform draw rarely
  or almost never hits: the sign bit of an INT8 weight (500 trials) and bits 24
  to 31 of an INT32 bias (200 trials; these bits are 0.2 % of the memory).
  High-order bits are known to dominate the damage (Li et al., 2017).

A trial has a **major** impact when more than 1 % of the predictions change.

Out of scope in v0.1: upsets in activations, in the requantisation constants
(scales and zero points), in the code and in the datapath, and any radiation
testing.

## 4. The monitor

### 4.1 Signals

| Signal | Reads | Good at | Blind to | State on board |
|---|---|---|---|---|
| Radiometry | the raw chip: mean, standard deviation, minimum and maximum of each band (12 statistics), as the largest absolute z-score against the training chips | exposure, saturation, dead pixels | semantic novelty at normal exposure; parameter faults | 96 bytes |
| Mahalanobis | the 128 pooled features: distance to the nearest class mean with a shared covariance (Lee et al., 2018, without input pre-processing) | inputs that look new to the network | inputs the network maps to typical features | 66,560 bytes in float32 (33,280 in float16) |
| Feature bounds | the same 128 features: summed exceedance of per-channel [min, max] bounds learnt on the training chips | features pushed out of range by a fault | faults that keep features within range | 1,024 bytes (256 as 8-bit codes) |
| Layer bounds (full monitor only) | the per-channel maximum of each convolution's UINT8 output (320 channels) | saturated or dead channels anywhere in the network | small perturbations | 640 bytes |

The bounds are those of range restriction (Chen et al., 2021, "Ranger"), used
here to detect rather than to clip. None of these signals needs a reference
copy of the weights.

For comparison, the evaluation also runs two **reference-based** integrity
checks that a flight system would combine with the monitor: a CRC-32 per
parameter tensor, recomputed by a memory scrub (RADAR uses checksums over groups
of weights in the same way against deliberate bit flips; Li et al., 2021), and
**golden inputs**, K stored
chips whose INT8 logits are compared bit for bit (integer inference is
deterministic, so any difference means the computation changed). And it runs
the usual output-based baselines, maximum softmax probability (Hendrycks and
Gimpel, 2017), max logit (Hendrycks et al., 2022) and energy (Liu et al., 2020),
and deep k-nearest neighbours (Sun et al., 2022), which stores 10,000 training features
and is kept as a reference only. These five baselines are among the methods
implemented in DEEL's oodeel library; Nerve Edge Decision re-implements them in NumPy so
that they run on the outputs of the INT8 graph.

### 4.2 Fusion

Each component score *s* becomes a conformal p-value against its own scores on
the clean validation chips, p = (#{v ≥ s} + 1) / (n + 1) (Bates et al., 2023).
The Nerve Edge Decision score is −log of the smallest p-value. Thresholding it at −log α
raises an alarm when, and only when, one component's p-value falls below α,
that is when its score exceeds a fixed threshold read off its own clean
validation scores. On board, the fusion therefore costs one comparison per
component and no p-value computation. Nothing is learnt in the fusion, and
nothing is tuned on test data.

### 4.3 Lite and full

* **Lite**: radiometry, Mahalanobis and feature bounds. It reads only the input
  chip and the pooled features, which the network computes anyway; the graph
  gains one DequantizeLinear node and one output.
* **Full**: lite plus the layer bounds, read by five ReduceMax taps on the
  convolution outputs.

### 4.4 Windows, for faults that persist

A corrupted weight stays corrupted until the memory is scrubbed, so every
inference carries some evidence. The online detectors average their score over
windows of W consecutive inferences (W = 1, 8 or 32) and compare the mean with
a threshold set for 1 % false alarms per window on the clean model. For the
fault campaigns these thresholds are calibrated on the clean test chips that the
campaign does not use, as a commissioning phase on in-flight data would do. The
results also show what happens when thresholds are carried over from the
validation scenes instead: window averaging magnifies small differences between
scenes.

## 5. Protocol

| Data | Used for |
|---|---|
| Training split | training; fitting radiometry, Mahalanobis, bounds and k-NN |
| Validation split | choosing the epoch; calibrating the fusion; golden inputs |
| Test split | every reported number; window thresholds of the fault campaigns come from test chips disjoint from the campaign chips |

* One seed. On a given machine and set of package versions, every stage is
  deterministic except timings, and one command (`make all`) reproduces every
  number.
* AUROC: probability that a shifted chip scores higher than a clean one, ties
  counting one half. FPR95: share of shifted chips still accepted when 95 % of
  clean chips are accepted. Proportions carry 95 % Wilson intervals.
* **Development history, stated plainly.** The corruption set and its severities
  were fixed before any result. A preliminary one-epoch run showed that
  feature-space detectors miss exposure changes: on mild low light the
  Mahalanobis AUROC was 0.04, because darker chips map to features that look
  more typical, not less. The radiometry signal and the fusion were added after
  that run, so the corruption benchmark is not a blind test of the final design.
  The harmful-shift criterion, the window sizes and the false-alarm targets were
  fixed before the final run.

## 6. Results

INT8 model, seed 0. Full tables in [`results/summary.md`](results/summary.md).

### 6.1 Quantisation

On 6,411 test chips from 324 scenes never seen in training, the INT8 network
reaches 98.81 % accuracy against 98.86 % in FP32 (−0.05 points); the two agree
on 99.95 % of the chips. The graph's parameters shrink from 558,728 to 144,874
bytes, and batch-1 latency on the host CPU drops from 234 to 122 µs.

### 6.2 Unusual inputs

| Mean AUROC | Nerve lite | Nerve full | Radiometry | Mahalanobis | MSP (best softmax-based) | Deep k-NN |
|---|---:|---:|---:|---:|---:|---:|
| 10 harmful shifts | 0.782 | **0.820** | 0.795 | 0.630 | 0.437 | 0.553 |
| All 16 shifts | 0.678 | **0.705** | 0.683 | 0.592 | 0.464 | 0.547 |
| Far OOD (CIFAR-10) | 0.891 | **0.898** | 0.893 | 0.875 | 0.291 | 0.727 |

What the numbers say:

1. **Scores built on the softmax output do worse than chance.** On the harmful
   shifts and on CIFAR-10 alike, corrupted or foreign inputs get *more*
   confident outputs than clean chips. A monitor that only reads the output would
   wave them through.
2. **Input-space and feature-space signals fail on different faults.**
   Mahalanobis catches strong stripes (0.82) but rates darker chips as more
   typical than clean ones (0.15 on mild low light); radiometry, 96 bytes of
   statistics about the raw chip, catches the exposure faults (0.88 to 0.99) but
   not stripes (0.62). The lite fusion keeps both kinds of coverage and pays a
   little for it: against radiometry alone it gains 0.14 on strong stripes and
   0.07 on a strong cloud, and loses up to 0.09 on the other eight harmful
   shifts, where the other components add their own false alarms. On the
   harmful mean the two are on par (0.782 against 0.795): this benchmark is
   dominated by photometric faults, which radiometry is built for.
3. **The full variant is the best detector overall** (0.820): its layer bounds
   see dead pixels (0.95 and 1.00) and strong noise that the pooled features
   average away.
4. **Two harmful shifts are barely detected by any signal**: strong blur (best
   AUROC 0.55) and a strong cloud (0.64).
5. **Quantisation does not degrade the monitor**: INT8 minus FP32 AUROC of the
   lite score is −0.001 on average over the 17 conditions (worst −0.007).
6. **Single chips are hard.** When 95 % of clean chips are accepted, the lite
   score still accepts 79 % of the chips of a harmful shift on average (FPR95).
   A degraded sensor, however, degrades every chip that follows; detection over
   windows of chips, as done below for bit flips, was not evaluated for inputs
   in v0.1 (Section 9).

### 6.3 Bit flips

| Bits flipped at random | Trials | Some prediction changes | More than 1 % change (major) | Major among trials sparing the biases |
|---:|---:|---:|---:|---:|
| 1 | 1,500 | 1.3 % | 0.1 % (1) | 0 of 1,489 |
| 4 | 300 | 10.3 % | 1.7 % (5) | 0 of 281 |
| 16 | 300 | 25.0 % | 2.0 % (6) | 0 of 259 |
| 64 | 300 | 48.3 % | 12.3 % (37) | 0 of 173 |
| 256 | 300 | 87.7 % | 39.0 % (117) | 0 of 29 |

| Targeted single flip | Trials | Major | Worst accuracy | Golden input (K = 1) on major | Layer bounds, single inference, on major |
|---|---:|---:|---:|---:|---:|
| Sign bit of an INT8 weight | 500 | 0 | 98.8 % | – | – |
| Bits 24–31 of an INT32 bias | 200 | 80 (40 %) | 64.8 % | 100 % | 98.8 % |

What the numbers say:

1. **The INT8 weights tolerate random upsets.** 98.7 % of single flips change no
   prediction, and none of the 2,231 uniform trials whose flips all fell in the
   weights had a major impact, with up to 256 flipped bits (about 32 of them
   sign bits). Flipping the sign bit of one weight never had one either (0 of
   500). With per-channel symmetric quantisation, a flipped weight can at most
   reach about the magnitude of the largest weight of its channel, and it is one
   input among 27 to 576 of a convolution.
2. **The INT32 biases carry the risk.** All 166 major-impact trials flipped at
   least one bias bit. A flip in bits 24 to 31 adds or removes at least 2²⁴ to a
   channel's accumulator: the channel saturates or goes dark for every input.
   Through conv1 to conv4 this has a major impact in 52 to 85 % of cases; in
   conv5, whose channels feed the pooled features directly, in none of 85
   trials. The 1,288 bytes of biases are 0.9 % of the parameter memory; every
   major impact in these campaigns involved them and none occurred without them,
   so they are the first memory to protect (ECC, or a triple copy for under
   4 KB).
3. **Detection.** A CRC-32 scrub detects every flip in the memory it covers, by
   construction, harmful or not. One golden input detected every major-impact
   trial. Without any reference, the layer bounds flag 98.8 % of inferences under
   the harmful bias flips, and they stay quiet when nothing happens: on the
   sign-bit trials, which changed almost nothing, they fire on 1.1 % of
   inferences, the rate measured on the clean model. The lite monitor's feature
   bounds flag 88 % of 32-inference windows under the harmful bias flips.
4. **Thresholds do not transfer freely between scenes.** Set on the validation
   scenes, the window thresholds of Mahalanobis and of the feature bounds give
   5 % false alarms per 32-inference window on the test scenes instead of 1 %:
   averaging over a window shrinks the noise, not the small shift between scenes.
   A flight system would recalibrate them on in-flight data, as the campaign
   does on held-out test chips.

The checks are complementary rather than competing:

| Check | Needs | Catches | When |
|---|---|---|---|
| CRC-32 scrub | 48 bytes of reference CRCs, a scrub schedule | every flip in the covered memory, harmful or not | at the next scrub |
| Golden inputs | 9,664 bytes for K = 8, K extra inferences per check | any change of the computation, memory or datapath | at the next check |
| Layer bounds (full) | 640 bytes, no reference | harmful saturated or dead channels | at the next inference |
| Feature bounds and Mahalanobis (lite) | 67,692 bytes, no reference | harmful faults that distort the features | within a window |

## 7. Costs

| INT8, host CPU, one thread | Network only | With lite monitor | With full monitor |
|---|---:|---:|---:|
| Latency at batch 1 (µs) | 121.6 | 123.4 | 173.7 |
| Per chip at batch 512 (µs) | 81.7 | 79.8 | 113.5 |
| Monitor state (bytes) | – | 67,692 (33,644 with float16) | 68,332 |
| Decision arithmetic per chip | – | 33,024 multiply-accumulates, one pass over 1,200 input values, 256 bound checks | plus 640 bound checks |

* The lite monitor's effect on latency is within measurement noise: it reads
  only the input and the pooled features, which the network computes anyway.
  Its decision step is about 0.3 % of the network's 11.4 million
  multiply-accumulates. In the NumPy reference implementation it takes 10.7 µs
  per chip over a batch of 512, and 67 µs at batch 1, where Python call overhead
  dominates.
* The full monitor's +43 % at batch 1 (+39 % per chip at batch 512) is the cost
  of five separate ReduceMax operators in ONNX Runtime, which re-read the 41,600
  values the convolutions write. Computed inside the convolution kernels, the
  maxima would be 41,600 comparisons per chip, 0.36 % of the network's
  multiply-accumulates; this is not measured here.
* Almost all of the lite state (65,536 of 67,692 bytes) is the 128 × 128
  precision matrix of Mahalanobis.
* Reference-based checks: a full CRC-32 scrub of the 140,648-byte parameter
  memory takes 66 µs on the host; checking one golden input takes one inference
  (127 µs), eight take 720 µs.

## 8. Limitations

* One dataset, one architecture, one seed, small 20 × 20 chips. With one seed,
  differences of a few hundredths of AUROC between detectors are not
  established.
* Simulated corruptions. Real degradations are subtler and correlated across a
  scene; the windows assume shuffled inputs, while a real stream is spatially
  correlated, which would raise the false-alarm rate of window detectors.
* Two harmful shifts, strong blur and a strong cloud, are detected by no signal
  here.
* The radiometry signal and the fusion were added after a preliminary run on
  the same corruption set (Section 5).
* Fault model restricted to the parameter memory (Section 3.2), and the
  conclusion about biases holds for this network and these campaigns.
* Timings come from ONNX Runtime on a cloud x86 CPU, not from an embedded
  target. Operation counts and memory sizes transfer; microseconds do not.
* The Mahalanobis precision matrix weighs 64 KiB in float32, about half the
  network's parameter memory. Float16 halves it; a low-rank or diagonal
  variant would cut it further and is not evaluated here.
* ECSS-E-HB-40-02A is cited for its safety-cage concept only.

## 9. Next steps

1. Detect persistent sensor faults over windows of chips, as done here for bit
   flips, and test a sharpness statistic for blur, on corruptions not used so
   far.
2. Port the INT8 network and the lite monitor to an embedded flow (a
   microcontroller with CMSIS-NN, or an FPGA flow such as FINN) and measure on
   the target.
3. Compute the per-channel maxima inside the convolution kernels instead of
   with separate operators.
4. Several seeds, a second dataset (Sentinel-2), real degraded imagery.
5. Faults in activations and requantisation constants; ECC on the biases
   against the cost of the monitor.
6. Map the evidence produced here to the activities of ECSS-E-HB-40-02A.

## References

* Bates, S., Candès, E., Lei, L., Romano, Y., Sesia, M. (2023). Testing for
  outliers with conformal p-values. *Annals of Statistics* 51(1).
* Chen, Z., Li, G., Pattabiraman, K. (2021). A low-cost fault corrector for deep
  neural networks through range restriction. *IEEE/IFIP DSN 2021*.
* ECSS (2024). ECSS-E-HB-40-02A, *Machine learning handbook*, 15 November 2024.
* Giuffrida, G. et al. (2022). The Φ-Sat-1 mission: the first on-board deep
  neural network demonstrator for satellite Earth observation. *IEEE
  Transactions on Geoscience and Remote Sensing* 60.
* Hammell, R. PlanesNet: planes in satellite imagery.
  github.com/rhammell/planesnet (CC BY-SA 4.0).
* Hendrycks, D., Gimpel, K. (2017). A baseline for detecting misclassified and
  out-of-distribution examples in neural networks. *ICLR 2017*.
* Hendrycks, D. et al. (2022). Scaling out-of-distribution detection for
  real-world settings. *ICML 2022*.
* Krizhevsky, A. (2009). Learning multiple layers of features from tiny images.
  Technical report, University of Toronto.
* Lee, K., Lee, K., Lee, H., Shin, J. (2018). A simple unified framework for
  detecting out-of-distribution samples and adversarial attacks. *NeurIPS 2018*.
* Li, G., Hari, S. K. S., Sullivan, M., Tsai, T., Pattabiraman, K., Emer, J.,
  Keckler, S. W. (2017). Understanding error propagation in deep learning neural
  network (DNN) accelerators and applications. *SC17*.
* Li, J., Rakin, A. S., He, Z., Fan, D., Chakrabarti, C. (2021). RADAR: run-time
  adversarial weight attack detection and accuracy recovery. *DATE 2021*.
* Liu, W., Wang, X., Owens, J., Li, Y. (2020). Energy-based out-of-distribution
  detection. *NeurIPS 2020*.
* Sun, Y., Ming, Y., Zhu, X., Li, Y. (2022). Out-of-distribution detection with
  deep nearest neighbors. *ICML 2022*.
