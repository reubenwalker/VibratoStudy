import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, get_window

# -----------------------------
# Parameters
# -----------------------------
fs = 48000
T = 2.0
t = np.linspace(0, T, int(fs*T), endpoint=False)

f0 = 330
vib_rate = 5.8
vib_extent_cents = 70

# Convert vibrato extent to FM ratio
vib_ratio = 2**(vib_extent_cents/1200) - 1

# Instantaneous frequency
inst_freq = f0 * (1 + vib_ratio * np.sin(2*np.pi*vib_rate*t))

# Integrate to phase
phase = 2*np.pi * np.cumsum(inst_freq) / fs

# -----------------------------
# Build harmonic signal
# -----------------------------
H = 20
amps = 1 / np.arange(1, H+1)   # −12 dB/oct style

signal = np.zeros_like(t)
for h in range(1, H+1):
    signal += amps[h-1] * np.sin(h * phase)

# -----------------------------
# Spectrogram (sharp version)
# -----------------------------
# Smaller FFT window → less blur
nperseg = 2048
noverlap = nperseg // 2
win = get_window("hann", nperseg)

f, tt, Sxx = spectrogram(signal, fs=fs, window=win,
                         nperseg=nperseg, noverlap=noverlap,
                         scaling='density', mode='magnitude')

Sxx_db = 20 * np.log10(Sxx + 1e-12)

# -----------------------------
# Plot (0–5000 Hz)
# -----------------------------
plt.figure(figsize=(12, 8))
plt.pcolormesh(tt, f, Sxx_db, shading='auto', cmap='inferno')

plt.ylim(0, 5000)
plt.clim(np.max(Sxx_db)-60, np.max(Sxx_db))   # dynamic range

plt.title("Spectrogram of Harmonic Vibrato Signal (70 cents, 5.8 Hz)\nZoom 0–5000 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")

plt.tight_layout()
plt.show()
