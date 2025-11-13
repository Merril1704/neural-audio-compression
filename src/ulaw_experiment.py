"""
Compare uniform N-bit quantization vs μ-law (mu-law) companding + N-bit quantization
for speech audio, and add simple ADPCM baselines. Reports SNR, segmental SNR, STOI, PESQ,
and estimates achievable compressed bitrate via symbol entropy. Supports CSV export
for RD plotting.

Usage (Windows cmd):
    python -m src.ulaw_experiment --wav examples\\speech.wav --bits 8 --sr 16000 --save out

If no --wav is provided, generates a dummy 5 s mono signal at --sr.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import torch

# Optional metrics libraries
try:
    from pystoi.stoi import stoi  # type: ignore
except Exception:
    stoi = None  # type: ignore

try:
    from pesq import pesq as pesq_metric  # type: ignore
except Exception:
    pesq_metric = None  # type: ignore


def _try_import_torchaudio():
    try:
        import torchaudio  # type: ignore
        return torchaudio
    except Exception:
        return None


def _try_import_soundfile():
    try:
        import soundfile as sf  # type: ignore
        return sf
    except Exception:
        return None


def load_audio_mono(wav_path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Load audio as mono, resample to target_sr, return (1, 1, T) tensor in [-1, 1].
    """
    ta = _try_import_torchaudio()
    if ta is not None:
        try:
            x, sr = ta.load(wav_path)  # (C, T)
            if x.shape[0] > 1:
                x = torch.mean(x, dim=0, keepdim=True)
            if sr != target_sr:
                try:
                    x = ta.functional.resample(x, sr, target_sr)
                except Exception:
                    x = ta.transforms.Resample(sr, target_sr)(x)
                sr = target_sr
            x = x.float().clamp(-1.0, 1.0)
            return x.unsqueeze(0), sr  # (1, 1, T)
        except Exception:
            # torchaudio couldn't decode (e.g., FLAC requires torchcodec/ffmpeg). Fall back to soundfile.
            pass

    # fallback
    sf = _try_import_soundfile()
    if sf is None:
        raise ImportError("Install torchaudio or soundfile to load audio files (e.g., FLAC/WAV).")
    data, sr = sf.read(wav_path, always_2d=True)  # (T, C)
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)
    data = data.astype(np.float32)
    # resample if needed using numpy/polyphase approx if scipy missing
    if sr != target_sr:
        try:
            from scipy import signal as sps  # type: ignore
            from math import gcd
            g = gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            data = sps.resample_poly(data, up=up, down=down, axis=0)
            sr = target_sr
        except Exception:
            raise ImportError("Resampling required but scipy not installed.")
    x = torch.from_numpy(data.T).unsqueeze(0)  # (1, 1, T)
    x = x.clamp(-1.0, 1.0)
    return x, sr


def ulaw_compand(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """
    x in [-1, 1] -> y in [-1, 1]
    y = sign(x) * ln(1 + mu*|x|) / ln(1 + mu)
    """
    mu = float(mu)
    x = torch.nan_to_num(x, nan=0.0).clamp(-1.0, 1.0)
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / math.log1p(mu)


def ulaw_expand(y: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """
    y in [-1, 1] -> x_hat in [-1, 1]
    x = sign(y) * (1/mu) * ( (1+mu)^{|y|} - 1 )
    Implement with exp to avoid pow with floats inconsistencies.
    """
    mu = float(mu)
    y = torch.nan_to_num(y, nan=0.0)
    return torch.sign(y) * (torch.expm1(torch.abs(y) * math.log1p(mu)) / mu)


def quantize_uniform(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Uniform quantize from [-1, 1] into 2^bits levels -> dequantized back to [-1, 1].
    Returns the dequantized reconstruction x_hat.
    """
    levels = 2 ** bits
    # Map [-1,1] to [0, levels-1]
    q = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    q = torch.round(q * (levels - 1))
    # Dequantize
    x_hat = (q / (levels - 1)) * 2.0 - 1.0
    return x_hat


def quantize_with_ulaw(x: torch.Tensor, bits: int, mu: int = 255) -> torch.Tensor:
    y = ulaw_compand(x, mu=mu)
    y_hat = quantize_uniform(y, bits)
    x_hat = ulaw_expand(y_hat, mu=mu)
    return x_hat


def _levels_and_centers(bits: int) -> Tuple[int, torch.Tensor]:
    L = 2 ** bits
    centers = torch.linspace(-1.0, 1.0, steps=L)
    return L, centers


def quantize_indices_uniform(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (indices, x_hat) for uniform quantization of x in [-1,1].
    indices: int64 tensor with values in [0, L-1]
    x_hat: dequantized reconstruction
    """
    L, centers = _levels_and_centers(bits)
    # sanitize input to avoid NaN -> long min
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    q = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    idx = torch.round(q * (L - 1)).to(torch.int64)
    idx = idx.clamp(0, L - 1)
    # Use a flattened index to avoid advanced-index edge-cases that can trigger
    # internal broadcasts/expand calls in some torch versions/devices.
    centers_dev = centers.to(x.device)
    flat_idx = idx.flatten()
    if flat_idx.numel() == 0:
        # empty input -> return matching empty tensors
        x_hat = idx.new_empty(idx.shape, dtype=centers_dev.dtype)
    else:
        vals = centers_dev[flat_idx]
        x_hat = vals.view(idx.shape)
    return idx, x_hat


def entropy_bits_per_symbol(indices: torch.Tensor) -> float:
    """Shannon entropy (bits/symbol) from integer index tensor."""
    flat = indices.flatten().cpu().numpy()
    if flat.size == 0:
        return 0.0
    hist = np.bincount(flat)
    p = hist[hist > 0].astype(np.float64)
    p /= p.sum()
    H = -np.sum(p * np.log2(p))
    return float(H)


def adpcm_encode_decode(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Simple non-adaptive ADPCM baseline with first-order predictor.
    - Predictor: x_pred[n] = x_hat[n-1]
    - Residual r = x - x_pred
    - Uniformly quantize residual in [-rmax, rmax] where rmax = max|r| over clip
    - Reconstruct sample-by-sample

    Returns reconstructed x_hat.
    """
    x = x.squeeze(0).squeeze(0)  # (T,)
    T = x.shape[0]
    x_hat = torch.zeros_like(x)
    prev = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    r = torch.empty_like(x)
    # First residual
    if T > 0:
        r[0] = x[0] - prev
    for n in range(1, T):
        r[n] = x[n] - x_hat[n-1]
    rmax = torch.max(torch.abs(r))
    rmax = torch.nan_to_num(rmax, nan=0.0) + 1e-8
    # Normalize residuals to [-1,1]
    r_norm = torch.clamp(torch.nan_to_num(r / rmax, nan=0.0), -1.0, 1.0)
    # Quantize residuals using arithmetic (avoid advanced indexing that may
    # trigger expand/broadcast issues on some torch builds/devices).
    L = 2 ** bits
    q = torch.clamp((r_norm + 1.0) * 0.5, 0.0, 1.0)
    idx = torch.round(q * (L - 1)).to(torch.int64).clamp(0, L - 1)
    # dequantize normalized residuals back to [-1,1]
    r_hat_norm = (idx.to(r_norm.dtype) / (L - 1)) * 2.0 - 1.0
    r_hat = r_hat_norm * rmax
    # Reconstruct
    if T > 0:
        x_hat[0] = prev + r_hat[0]
    for n in range(1, T):
        x_hat[n] = x_hat[n-1] + r_hat[n]
    return x_hat.unsqueeze(0).unsqueeze(0)


def adpcm_ulaw_encode_decode(x: torch.Tensor, bits: int, mu: int = 255) -> torch.Tensor:
    """
    μ-law compand the signal, run ADPCM as above in μ-law domain, then expand.
    """
    y = ulaw_compand(x, mu=mu)
    y_hat = adpcm_encode_decode(y, bits)
    x_hat = ulaw_expand(y_hat, mu=mu)
    return x_hat


def snr_db(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-12) -> float:
    """Robust SNR in dB; guards against zero/NaN/Inf to avoid log domain errors."""
    x = torch.nan_to_num(x, nan=0.0)
    x_hat = torch.nan_to_num(x_hat, nan=0.0)
    num = float(torch.sum(x ** 2).item())
    den = float(torch.sum((x - x_hat) ** 2).item())
    if not math.isfinite(num) or num <= 0.0:
        num = eps
    if not math.isfinite(den) or den <= 0.0:
        den = eps
    ratio = num / den
    if not math.isfinite(ratio) or ratio <= 0.0:
        ratio = eps
    return 10.0 * math.log10(ratio)


def segmental_snr_db(x: torch.Tensor, x_hat: torch.Tensor, frame_size: int = 320, hop: int = 160, eps: float = 1e-12) -> float:
    """
    Segmental SNR over short frames (e.g., 20 ms at 16 kHz). Returns average over frames.
    x, x_hat shapes: (1, 1, T)
    """
    x = torch.nan_to_num(x).squeeze(0).squeeze(0)
    xh = torch.nan_to_num(x_hat).squeeze(0).squeeze(0)
    T = x.shape[-1]
    if T <= 0:
        return float('nan')
    ssnrs: list[float] = []
    for start in range(0, max(T - frame_size + 1, 1), hop):
        end = min(start + frame_size, T)
        xe = x[start:end]
        xhe = xh[start:end]
        num = float(torch.sum(xe ** 2).item())
        den = float(torch.sum((xe - xhe) ** 2).item())
        if not math.isfinite(num):
            num = 0.0
        if not math.isfinite(den):
            den = 0.0
        if num <= eps and den <= eps:
            continue
        num = max(num, eps)
        den = max(den, eps)
        ratio = num / den
        if not math.isfinite(ratio) or ratio <= 0.0:
            continue
        ssnrs.append(10.0 * math.log10(ratio))
    if not ssnrs:
        return float('nan')
    return float(np.mean(ssnrs))


def compute_stoi(x: torch.Tensor, x_hat: torch.Tensor, sr: int) -> Optional[float]:
    if stoi is None:
        return None
    ref = x.squeeze().cpu().numpy()
    deg = x_hat.squeeze().cpu().numpy()
    try:
        return float(stoi(ref, deg, sr, extended=False))
    except Exception:
        return None


def compute_pesq(x: torch.Tensor, x_hat: torch.Tensor, sr: int) -> Optional[float]:
    if pesq_metric is None:
        return None
    # PESQ supports 8000 (nb) and 16000 (wb). Choose mode accordingly.
    mode = None
    if sr == 16000:
        mode = 'wb'
    elif sr == 8000:
        mode = 'nb'
    else:
        return None
    ref = x.squeeze().cpu().numpy()
    deg = x_hat.squeeze().cpu().numpy()
    try:
        return float(pesq_metric(sr, ref, deg, mode))
    except Exception:
        return None


def save_wav(path: str, x: torch.Tensor, sr: int) -> None:
    """Save WAV with robust fallbacks.

    Prefers torchaudio if available, but falls back to soundfile if saving fails
    (e.g., when torchcodec isn't installed). Accepts input shaped (1, 1, T).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    ta = _try_import_torchaudio()
    if ta is not None:
        try:
            # torchaudio expects (C, T)
            ta.save(path, x.squeeze(0), sample_rate=sr)
            return
        except Exception:
            # Fall back to soundfile if torchaudio cannot save (e.g., torchcodec missing)
            pass

    sf = _try_import_soundfile()
    if sf is not None:
        data = x.squeeze(0).cpu().numpy()  # (1, T) or (C, T)
        # If mono (1, T) -> (T,), else soundfile accepts (T, C) so transpose if needed
        if data.ndim == 2 and data.shape[0] == 1:
            data = data[0]
        elif data.ndim == 2 and data.shape[0] > 1:
            data = data.T  # (T, C)
        sf.write(path, data, sr)
        return
    # If neither backend is available, do nothing (print hint)
    print("Warning: Could not save WAV (neither torchaudio nor soundfile available)")


def evaluate_condition(
    x: torch.Tensor,
    sr: int,
    method: str,
    bits: int,
    mu: int = 255,
) -> Dict[str, Any]:
    """Run a single condition and compute metrics + entropy bitrate.

    Methods:
      - uniform_pcm
      - ulaw_pcm
      - adpcm_uniform
      - adpcm_ulaw
    """
    assert method in {"uniform_pcm", "ulaw_pcm", "adpcm_uniform", "adpcm_ulaw"}
    # Reconstruct
    if method == "uniform_pcm":
        idx, x_hat = quantize_indices_uniform(x, bits)
        idx_stream = idx
    elif method == "ulaw_pcm":
        y = ulaw_compand(x, mu=mu)
        idx, y_hat = quantize_indices_uniform(y, bits)
        x_hat = ulaw_expand(y_hat, mu=mu)
        idx_stream = idx
    elif method == "adpcm_uniform":
        # Encode residual indices internally in adpcm function? For entropy, recompute r indices here for simplicity
        x_hat = adpcm_encode_decode(x, bits)
        # Approximate symbol stream by re-deriving residual indices from the above pass
        x1 = x.squeeze(0).squeeze(0)
        xh1 = x_hat.squeeze(0).squeeze(0)
        T = x1.shape[0]
        r = torch.empty_like(x1)
        r[0] = x1[0]
        for n in range(1, T):
            r[n] = x1[n] - xh1[n-1]
        rmax = torch.max(torch.abs(r))
        rmax = torch.nan_to_num(rmax, nan=0.0) + 1e-8
        r_norm = torch.clamp(torch.nan_to_num(r / rmax, nan=0.0), -1.0, 1.0)
        # Quantize residual indices arithmetically to avoid indexing edge-cases
        L = 2 ** bits
        q = torch.clamp((r_norm + 1.0) * 0.5, 0.0, 1.0)
        idx_stream = torch.round(q * (L - 1)).to(torch.int64).clamp(0, L - 1)
    else:  # adpcm_ulaw
        x_hat = adpcm_ulaw_encode_decode(x, bits, mu=mu)
        # Derive residual indices in μ-law domain
        y = ulaw_compand(x, mu=mu).squeeze(0).squeeze(0)
        yh = ulaw_compand(x_hat, mu=mu).squeeze(0).squeeze(0)
        T = y.shape[0]
        r = torch.empty_like(y)
        r[0] = y[0]
        for n in range(1, T):
            r[n] = y[n] - yh[n-1]
        rmax = torch.max(torch.abs(r))
        rmax = torch.nan_to_num(rmax, nan=0.0) + 1e-8
        r_norm = torch.clamp(torch.nan_to_num(r / rmax, nan=0.0), -1.0, 1.0)
        L = 2 ** bits
        q = torch.clamp((r_norm + 1.0) * 0.5, 0.0, 1.0)
        idx_stream = torch.round(q * (L - 1)).to(torch.int64).clamp(0, L - 1)

    snr = snr_db(x, x_hat)
    segsnr = segmental_snr_db(x, x_hat)
    stoi_v = compute_stoi(x, x_hat, sr)
    pesq_v = compute_pesq(x, x_hat, sr)

    # Bitrates
    nominal_bps = bits * sr  # mono, symbols per sample = 1
    H = entropy_bits_per_symbol(idx_stream)
    entropy_bps = H * sr

    return {
        "method": method,
        "bits": bits,
        "mu": mu,
        "sr": sr,
        "snr_db": snr,
        "segsnr_db": segsnr,
        "stoi": stoi_v,
        "pesq": pesq_v,
        "nominal_bps": nominal_bps,
        "entropy_bps": entropy_bps,
    }


def main():
    p = argparse.ArgumentParser(description="μ-law vs uniform N-bit quantization with ADPCM, metrics, and bitrate estimates")
    p.add_argument("--wav", type=str, default=None, help="Path to a single audio file (e.g., .flac/.wav; ideally speech)")
    p.add_argument("--wav-dir", type=str, default=None, help="Directory of audio files to batch process (e.g., dataset)")
    p.add_argument("--pattern", type=str, default="*.flac", help="Glob pattern(s) when using --wav-dir (e.g., *.flac or \"*.flac,*.wav\"). Supports recursive patterns like **/*.flac")
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate (speech typical: 16000)")
    p.add_argument("--seconds", type=float, default=5.0, help="Duration for dummy audio if no WAV")
    p.add_argument("--bits", type=int, nargs="+", default=[8], help="Quantization bits (e.g., 6 8 10)")
    p.add_argument("--mu", type=int, default=255, help="μ for companding")
    p.add_argument("--methods", type=str, nargs="+", default=["uniform_pcm", "ulaw_pcm", "adpcm_uniform", "adpcm_ulaw"],
                   help="Methods to run: uniform_pcm ulaw_pcm adpcm_uniform adpcm_ulaw")
    p.add_argument("--save", type=str, default=None, help="Folder to save reconstructions for the last run condition")
    p.add_argument("--save-all", action="store_true", help="Save reconstructions for every processed condition into --save (or output/last_run_all if --save omitted)")
    p.add_argument("--csv", type=str, default=None, help="Path to write metrics CSV (e.g., output/metrics.csv)")
    args = p.parse_args()

    results: List[Dict[str, Any]] = []

    def run_one_source(x: torch.Tensor, sr: int, src_label: str):
        print(f"Input: {src_label} @ {sr} Hz, mu={args.mu}")
        for b in args.bits:
            for m in args.methods:
                try:
                    res = evaluate_condition(x, sr, m, b, mu=args.mu)
                except Exception as e:
                    print(f"Warning: failed {src_label} {m} {b}-bit: {e}")
                    continue
                results.append({"source": src_label, **res})
                # Brief printout
                stoi_str = f"{res['stoi']:.3f}" if res['stoi'] is not None else "n/a"
                pesq_str = f"{res['pesq']:.3f}" if res['pesq'] is not None else "n/a"
                print(f"{m:14s} {b:2d}-bit | SNR {res['snr_db']:.2f} dB | SegSNR {res['segsnr_db']:.2f} dB | "
                      f"STOI {stoi_str} | PESQ {pesq_str} | H {res['entropy_bps']/1000:.1f} kbps (nom {res['nominal_bps']/1000:.1f} kbps)")
                # Optionally save every reconstruction for this condition
                if args.save_all:
                    # Determine base folder
                    base_root = args.save if args.save is not None else os.path.join("output", "last_run_all")
                    # Use a safe source basename
                    try:
                        src_base = os.path.splitext(os.path.basename(str(src_label)))[0]
                    except Exception:
                        src_base = str(src_label)
                    out_dir = os.path.join(base_root, src_base)
                    os.makedirs(out_dir, exist_ok=True)
                    # Reconstruct the waveform for saving
                    try:
                        xh = None
                        if m == "uniform_pcm":
                            _, xh = quantize_indices_uniform(x, b)
                        elif m == "ulaw_pcm":
                            y = ulaw_compand(x, mu=args.mu)
                            _, yh = quantize_indices_uniform(y, b)
                            xh = ulaw_expand(yh, mu=args.mu)
                        elif m == "adpcm_uniform":
                            xh = adpcm_encode_decode(x, b)
                        else:  # adpcm_ulaw
                            xh = adpcm_ulaw_encode_decode(x, b, mu=args.mu)
                    except Exception as e:
                        print(f"Warning: could not reconstruct for saving {src_label} {m} {b}-bit: {e}")
                        xh = None
                    if xh is not None:
                        # Build filename; include mu for ulaw methods
                        if "ulaw" in m:
                            fname = f"{m}_mu{args.mu}_{b}bit.wav"
                        else:
                            fname = f"{m}_{b}bit.wav"
                        dest = os.path.join(out_dir, fname)
                        # avoid overwriting by finding a unique filename
                        base, ext = os.path.splitext(dest)
                        i = 1
                        unique_dest = dest
                        while os.path.exists(unique_dest):
                            unique_dest = f"{base}_{i}{ext}"
                            i += 1
                        try:
                            save_wav(unique_dest, xh, sr)
                            print("Saved:", os.path.abspath(unique_dest))
                        except Exception as e:
                            print(f"Warning: failed to save {unique_dest}: {e}")

    # Batch mode: directory of audio files
    if args.wav_dir:
        from pathlib import Path
        wav_dir = Path(args.wav_dir)
        # Support multiple patterns separated by comma/semicolon
        raw_patterns = [s.strip() for s in str(args.pattern).replace(";", ",").split(",") if s.strip()]
        files: List[str] = []
        for pat in raw_patterns:
            files.extend([str(p) for p in wav_dir.glob(pat)])
        files = sorted(set(files))
        if not files:
            pats_str = ", ".join(raw_patterns) if raw_patterns else str(args.pattern)
            print(f"No files matched in {args.wav_dir} for patterns: {pats_str}")
        for fp in files:
            try:
                x, sr = load_audio_mono(fp, target_sr=args.sr)
                src_label = fp
                run_one_source(x, sr, src_label)
            except Exception as e:
                print(f"Warning: skipping {fp}: {e}")
    else:
        # Single file or dummy
        if args.wav:
            x, sr = load_audio_mono(args.wav, target_sr=args.sr)
            src_label = args.wav
        else:
            T = int(args.seconds * args.sr)
            # Dummy speech-like: sum of a few bands with amplitude modulations
            t = torch.linspace(0, args.seconds, T)
            sig = 0.5*torch.sin(2*math.pi*200*t) + 0.3*torch.sin(2*math.pi*400*t) + 0.2*torch.sin(2*math.pi*800*t)
            env = 0.5*(1+torch.sin(2*math.pi*2*t))
            sig = (sig * env).unsqueeze(0).unsqueeze(0).float()
            x, sr = sig.clamp(-1, 1), args.sr
            src_label = f"dummy_{args.seconds}s_{args.sr}Hz"
        run_one_source(x, sr, src_label)

    # Save reconstructions for the last combination if requested
    if args.save:
        last = results[-1]
        b = last["bits"]
        m = last["method"]
        # Re-run the reconstruction to save audio
        if m == "uniform_pcm":
            _, xh = quantize_indices_uniform(x, b)
        elif m == "ulaw_pcm":
            y = ulaw_compand(x, mu=args.mu)
            _, yh = quantize_indices_uniform(y, b)
            xh = ulaw_expand(yh, mu=args.mu)
        elif m == "adpcm_uniform":
            xh = adpcm_encode_decode(x, b)
        else:
            xh = adpcm_ulaw_encode_decode(x, b, mu=args.mu)

        base = args.save
        os.makedirs(base, exist_ok=True)
        save_wav(os.path.join(base, "input.wav"), x, sr)
        save_wav(os.path.join(base, f"{m}_{b}bit.wav"), xh, sr)
        print("Saved reconstructions to:", os.path.abspath(base))

    # CSV export
    if args.csv:
        try:
            import pandas as pd  # type: ignore
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            df = pd.DataFrame(results)
            df.to_csv(args.csv, index=False)
            print("Wrote metrics CSV:", os.path.abspath(args.csv))
        except Exception as e:
            print("Warning: could not write CSV:", e)


if __name__ == "__main__":
    main()
