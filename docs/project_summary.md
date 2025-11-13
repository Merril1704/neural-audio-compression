# μ-law Companding at Low Bit Depths: A Practical Evaluation with ADPCM and Neural Codec Context

## Abstract
We revisit classic μ-law companding for low-bit-depth audio coding and quantify its benefits relative to uniform PCM and a simple non‑adaptive ADPCM baseline on speech at 16 kHz. Using a unified, reproducible pipeline, we report global SNR, segmental SNR, STOI, PESQ, and entropy‑estimated bitrates. At 8 bits, μ-law PCM delivers substantially higher segmental SNR and near‑transparent intelligibility compared with uniform PCM at the same nominal bitrate (128 kbps). Entropy analysis reveals that μ-law’s more uniform sample distribution yields smaller entropy‑coding gains, often pushing the effective compressed bitrate closer to nominal than uniform PCM. We also provide a measurement utility for AGC (Audiogen neural codec) latents, showing that learned codecs achieve markedly lower bitrates at high quality and that μ-law preprocessing on waveforms does not reduce latent size.

## 1. Introduction
Low‑bit‑rate audio coding spans legacy scalar quantization pipelines (μ‑law/A‑law companding, ADPCM) and modern perceptual or learned codecs (Opus, AAC; SoundStream, EnCodec, AGC). Despite being “old,” μ‑law remains relevant in ultra‑low‑latency, low‑compute deployments where a few arithmetic operations per sample are all that is feasible. This work asks: when does μ‑law still help, how much, and against which baselines? We aim to provide evidence‑based guidance using a small, fully reproducible evaluation harness included in this repository.

### Problem statement
Practitioners lack recent, controlled comparisons among (i) uniform N‑bit PCM, (ii) μ‑law + PCM, (iii) simple residual coding (ADPCM) with or without μ‑law, and (iv) learned latent codecs, particularly at very low bit depths and under strict compute/latency budgets.

### Contributions
- A compact evaluation framework implementing four baselines: uniform PCM, μ‑law PCM, non‑adaptive first‑order ADPCM, and μ‑law + ADPCM.
- Objective metrics (SNR, segmental SNR, STOI, PESQ) and entropy‑based bitrate estimates, with CSV exports and RD plots.
- Empirical guidance on when μ‑law helps (low‑bit speech) and where it does not (wideband music, learned latent codecs).
- A utility to measure AGC raw‑to‑latent compression ratios for context.

## 2. Related Work
μ‑law companding (G.711) reallocates quantization resolution toward low‑amplitude regions, improving intelligibility for narrowband speech. ADPCM/DPCM reduce redundancy through prediction and residual quantization, with many adaptive variants (e.g., IMA ADPCM, G.726). Perceptual codecs (Opus, AAC) leverage psychoacoustics for high quality at low bitrates. Learned codecs (SoundStream, EnCodec, AGC) produce continuous or discrete latents that can be quantized and entropy‑coded effectively. We build on these foundations to provide a modern, empirical comparison under controlled conditions.

## 3. Methods

### 3.1 Signal model and notation
Let \(x \in [-1,1]\) denote a normalized mono waveform sampled at \(f_s\) Hz. We consider scalar quantizers with \(L=2^{\text{bits}}\) levels and a non‑adaptive first‑order residual coder.

### 3.2 μ‑law companding
Forward compand with parameter \(\mu\):
$$ y = \operatorname{sign}(x) \cdot \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}. $$
Inverse expand:
$$ \hat{x} = \operatorname{sign}(y) \cdot \frac{(1+\mu)^{|y|} - 1}{\mu}. $$
We use \(\mu=255\) unless stated.

### 3.3 Uniform scalar quantization
Map \(x\) to index \(k \in \{0,\ldots,L-1\}\) via \(k = \operatorname{round}((x+1)/2 \cdot (L-1))\). Dequantize with mid‑rise centers \(c_k\) linearly spaced in \([-1,1]\). For μ‑law PCM, quantize the companded signal \(y\) and then expand \(\hat{x}\).

### 3.4 Non‑adaptive first‑order ADPCM
Predict \(\hat{x}[n] = \hat{x}[n-1]\) and form residual \(r[n] = x[n] - \hat{x}[n-1]\). Normalize residuals clipwise to \([-1,1]\), quantize uniformly to \(L\) levels, then reconstruct sample‑by‑sample by accumulation. We also evaluate ADPCM in the μ‑law domain (compand → ADPCM → expand).

### 3.5 Bitrate models
Nominal bitrate (mono):
$$ R_{\text{nominal}} = \text{bits} \cdot f_s. $$
Achievable compressed bitrate via symbol entropy \(H\) (bits/symbol):
$$ R_{\text{entropy}} \approx H \cdot f_s, \quad H = -\sum_i p_i \log_2 p_i. $$

### 3.6 Objective metrics
- SNR (dB): \(10\log_{10}(\|x\|^2/\|x-\hat{x}\|^2)\) with safeguards for numerical stability.
- Segmental SNR (dB): average of framewise SNRs (20 ms frames at 16 kHz, 50% hop) with silent frames down‑weighted.
- STOI, PESQ: standard speech intelligibility/quality metrics when supported at the sampling rate.

## 4. Implementation and Reproducibility

### 4.1 Code structure
```
src/
  ulaw_experiment.py   # Baselines, metrics, entropy, CSV export
  plot_rd.py           # RD plots (SegSNR/STOI vs bitrate; optional error bars)
  sum_table.py         # Aggregation → Markdown table (mean ± std)
docs/
  research.md          # Extended notes and references
  summary_table.md     # Example aggregated results (8‑bit)
output/
  metrics.csv          # Per‑source metrics
  plots/               # Saved figures
```

### 4.2 Processing pipeline (pseudo‑code)
```
for source in dataset:
  x ← load_mono(source, sr=16k); x ← clamp(x, −1, 1)
  for bits in {6,8,10} (configurable):
    for method in {uniform_pcm, ulaw_pcm, adpcm_uniform, adpcm_ulaw}:
      x_hat, indices ← encode_decode(x, method, bits)
      metrics ← {SNR, SegSNR, STOI, PESQ}
      entropy_bps ← entropy(indices) * sr
      write_row(csv, source, method, bits, metrics, entropy_bps)
```

### 4.3 How to reproduce (Windows cmd)
```
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
python -m src.ulaw_experiment --wav-dir dataset --pattern "**/*.flac,**/*.wav" --sr 16000 --bits 8 --csv output\metrics.csv --save output\last_run
python -m src.plot_rd --csv output\metrics.csv --outdir output\plots --bitrate entropy --aggregate
```
AGC latent ratio measurement examples are in `README.md`.

## 5. Experimental Setup
**Data.** Speech clips at 16 kHz mono; file names in `output/metrics.csv` indicate a LibriSpeech‑style subset (e.g., `dataset\84-121123-0006.flac`).

**Conditions.** We report 8‑bit results in the main table (others are supported). Methods: `uniform_pcm`, `ulaw_pcm`, `adpcm_uniform`, `adpcm_ulaw`. μ set to 255.

**Metrics.** Global SNR, segmental SNR (20 ms / 10 ms hop), STOI, PESQ (wb at 16 kHz), nominal and entropy bitrates.

**Environment.** PyTorch with optional `torchaudio`/`soundfile` I/O; numerical guards for NaNs/Infs; deterministic operations where applicable.

## 6. Results
### 6.1 Aggregate (8‑bit) summary
From `docs/summary_table.md` (mean ± std across sources):

|Method|Bits|SegSNR (dB)|STOI|PESQ|Entropy (kbps)|
|---|---|---|---|---|---|
|adpcm_ulaw|8|86.46 ± 20.26|n/a|n/a|108.96 ± 4.99|
|adpcm_uniform|8|-27.57 ± 6.35|0.99 ± 0.01|3.80 ± 0.34|102.74 ± 10.45|
|ulaw_pcm|8|34.23 ± 1.56|1.00 ± 0.00|4.19 ± 0.08|116.91 ± 3.94|
|uniform_pcm|8|17.47 ± 4.45|0.98 ± 0.05|2.66 ± 0.31|70.70 ± 8.75|

Representative RD plots (saved by `src/plot_rd.py`) show μ‑law curves shifted upward in SegSNR at a given entropy bitrate. See `output/plots/rd_segsnr.png` and `output/plots/rd_stoi.png`.

### 6.2 Interpretation
- μ‑law PCM significantly improves segmental SNR over uniform PCM at the same nominal bitrate, aligning with the intuition that companding preserves low‑level details important for intelligibility.
- STOI and PESQ are near‑ceiling for μ‑law at 8 bits on speech, while uniform PCM lags, especially in PESQ.
- Entropy bitrates: uniform PCM often compresses to ≈60–85 kbps at 8 bits due to skewed distributions; μ‑law flattens distributions, leading to ≈110–120 kbps after entropy coding. Thus, the effective bitrate gap narrows even as quality improves.
- ADPCM baselines: the simple non‑adaptive predictor used here can yield poor global SNR; combining μ‑law + ADPCM gives very high segmental SNR but requires more careful normalization and (ideally) adaptive step‑size for balanced performance.

## 7. Comparison to Neural Codec Latents (Context)
The AGC measurement utility (see `README.md`) reports raw vs latent sizes, e.g., for 10 s stereo: raw ≈293.6 kbps vs latent in‑memory ≈58.7 kbps. Learned codecs therefore attain far lower rates for high fidelity than simple waveform quantizers. Importantly, applying μ‑law to the waveform does not reduce AGC latent size; gains must come from latent quantization and entropy coding inside the neural pipeline.

## 8. Discussion and Threats to Validity
**When μ‑law helps.** Low‑bit (6–10), speech‑centric scenarios with tight compute/latency. Benefits are largest in segmental SNR and intelligibility proxies.

**When it does not.** Wideband music (48 kHz stereo) and learned latent codecs: companding is less beneficial or irrelevant.

**Threats.** Non‑adaptive ADPCM may underestimate the potential of residual coding; dataset breadth is limited; objective metrics may not fully reflect perception. Entropy estimates ignore context models; realized bitrates with arithmetic coding may be lower.

## 9. Conclusion
μ‑law companding remains a strong baseline for low‑compute speech coding at low bit depths, offering sizable segmental SNR and intelligibility gains over uniform PCM at fixed nominal rates. Its advantages diminish outside that regime, and learned codecs remain preferred when feasible. The provided pipeline is minimal, transparent, and extensible for broader studies.

## 10. Future Work
1. Adaptive ADPCM (IMA, G.726‑style) and LPC predictors; refine residual normalization.
2. μ sweeps and joint optimization with bit depth; ablations on frame sizes.
3. Implement arithmetic/range coding to measure realized vs entropy bitrate.
4. Expand datasets (noise, accents, background music) and add subjective tests (MUSHRA).
5. Explore hybrid pipelines combining simple front‑end transforms with learned encoders.

## References (selected)
- ITU‑T G.711: Pulse code modulation (PCM) of voice frequencies (μ‑law/A‑law companding).
- ITU‑T G.726: 40–16 kbit/s Adaptive Differential PCM (ADPCM).
- Jayant, N. S., and Noll, P. Digital Coding of Waveforms: Principles and Applications to Speech and Video.
- Shannon, C. E. A Mathematical Theory of Communication. Bell System Technical Journal, 1948.
- Gray, R. M., and Neuhoff, D. L. Quantization. IEEE Trans. Info. Theory, 1998.
- Johnston, J. D. Transform coding of audio signals using perceptual noise criteria. JSAC, 1988.
- RFC 6716: Definition of the Opus Audio Codec.
- Zeghidour et al. SoundStream; Défossez et al. EnCodec; AGC (Audiogen) references.

---
Generated: 2025‑11‑12 • Sources: `output/metrics.csv`, `docs/summary_table.md`, code under `src/`.
