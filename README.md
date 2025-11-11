# data_comp: AGC Compression Ratio Helper

This repo includes a tiny script to measure the compression ratio of the AGC audio codec by comparing raw audio tensor size to the encoded latent size.

## What it does
- Loads a pretrained AGC model (continuous or discrete)
- Loads a WAV file (resampled to 48 kHz) or generates a dummy waveform
- Encodes the audio to latent `z`
- Reports shapes, byte sizes, element ratios, and (approx) bitrates based on in-memory tensor sizes
- Optionally saves the latent to disk for inspection

Note: The computed "compressed bitrate" is based on the in-memory tensor dtype of the latent returned by the model. The theoretical bitrate can be lower if the latent is quantized or stored more compactly than the tensor representation.

## Quick start (Windows cmd)

First, create/activate a Python environment and install dependencies (PyTorch install method depends on your CUDA/CPU setup; see pytorch.org if needed):

```cmd
py -m pip install --upgrade pip
py -m pip install torch torchaudio soundfile librosa scipy
```

Now run the compression measurement on a WAV file:

```cmd
py -m src.measure_compression --wav examples\your_audio.wav --model Audiogen/agc-continuous
```

Or generate a dummy 10s stereo input:

```cmd
py -m src.measure_compression --seconds 10 --channels 2 --model Audiogen/agc-continuous
```

Use the discrete model instead:

```cmd
py -m src.measure_compression --wav examples\your_audio.wav --model Audiogen\agc-discrete
```

Optionally save the latent to disk to inspect file size:

```cmd
py -m src.measure_compression --wav examples\your_audio.wav --model Audiogen/agc-continuous --save-z output\latent.pt
```

### μ-law vs uniform quantization experiment

Run a small experiment that shows where μ-law companding helps at the same bit depth (e.g., 8‑bit) on speech-like audio. It reports SNR and segmental SNR gains and can save reconstructions.

```cmd
py -m src.ulaw_experiment --wav examples\speech.wav --bits 8 --sr 16000 --save output\ulaw
```

If you don’t have a speech WAV handy, generate a dummy 5-second signal:

```cmd
py -m src.ulaw_experiment --seconds 5 --bits 8 --sr 16000 --save output\ulaw_dummy
```

You should see μ-law produce higher SNR than uniform 8‑bit for speech at 16 kHz mono. For hi‑fi 48 kHz stereo music, μ‑law is less effective; use learned codecs or perceptual codecs instead.

## Inputs and assumptions
- Sample rate: 48 kHz expected by AGC; loader resamples if needed
- Input shape: (1, C, T)
- Outputs:
  - Audio shape and byte size
  - Latent shape and byte size
  - Elements ratio and bytes ratio
  - Approximate raw/compressed bitrate (kbps)

If you run into install issues for `torchaudio`, you can rely on `soundfile` + `librosa` or `scipy` for audio loading/resampling.

## Example output
```
Model: Audiogen/agc-continuous on cpu
Audio shape: (1, 2, 480000) dtype bytes: 4 size: 3.67 MB
Latent shape: (1, 32, 6000) dtype bytes: 4 size: 0.73 MB
Duration: 10.000 s
Elements ratio (audio/latent): 5.00x
Bytes ratio (audio/latent): 5.00x
Raw bitrate: 293.6 kbps
Compressed bitrate (latent in-memory): 58.7 kbps
```

## Notes
- If you want to compare file sizes on disk, use `--save-z` and check the file size of the `.pt` file. Be aware the serialization format has overhead compared to an ideal bitstream.
- For GPU acceleration, add `--device cuda` if a compatible CUDA-enabled PyTorch is installed.

## Publish to GitHub (Windows)

If you want this project in your GitHub repo (e.g., `neural-audio-compression`), run these from the project root (`D:\Projects\data_comp`):

```cmd
git init
git branch -M main
git add .
git commit -m "Initial import: μ-law/ADPCM experiments, plots, and docs"
git remote add origin https://github.com/<your-username>/neural-audio-compression.git
git push -u origin main
```

Optional: enable Git LFS before pushing large audio/models/plots:

```cmd
git lfs install
git add .gitattributes
git commit -m "Enable Git LFS for audio/models/plots"
git push
```

Replace `<your-username>` with your GitHub username. The `.gitignore` keeps local datasets (`dataset/`) and generated outputs (`output/`) out of version control by default.
