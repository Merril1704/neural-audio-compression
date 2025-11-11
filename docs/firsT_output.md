
## Ran just the audio codec 
Model: Audiogen/agc-continuous on cpu
Audio shape: (1, 2, 240000) dtype bytes: 4 size: 1.83 MB
Latent shape: (1, 32, 500) dtype bytes: 4 size: 62.50 KB
Latent shape: (1, 32, 500) dtype bytes: 4 size: 62.50 KB
Duration: 5.000 s
Elements ratio (audio/latent): 30.00x
Bytes ratio (audio/latent): 30.00x
Raw bitrate: 3072.0 kbps
Compressed bitrate (latent in-memory): 102.4 kbps


## paper approach 
(audio_codec) D:\Projects\data_comp>python -m src.ulaw_experiment --seconds 5 --bits 8 --sr 16000 --save output\ulaw_dummy
Input: dummy @ 16000 Hz, bits=8, mu=255
Uniform 8-bit:    SNR=41.22 dB, SegSNR=34.41 dB
μ-law + 8-bit:    SNR=37.89 dB, SegSNR=37.13 dB
Gain (μ-law over uniform):  ΔSNR=-3.34 dB, ΔSegSNR=2.72 dB
Saved reconstructions to: D:\Projects\data_comp\output\ulaw_dummy
