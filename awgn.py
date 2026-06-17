"""
Additive White Gaussian Noise (AWGN)

AWGN models the effect of random noise added to a signal in a channel.
The noise is:
  - Additive: signal + noise
  - White: flat power spectral density (all frequencies equally affected)
  - Gaussian: noise samples follow a normal distribution N(0, sigma^2)
"""

import numpy as np
import matplotlib.pyplot as plt


def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise to a signal at the specified SNR.

    Args:
        signal: Input signal (1-D array).
        snr_db: Desired signal-to-noise ratio in decibels.

    Returns:
        Noisy signal with the same shape as input.
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def compute_snr_db(signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """Return the actual SNR (dB) between a clean signal and its noisy version."""
    noise = noisy_signal - signal
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


def plot_awgn(signal: np.ndarray, noisy_signal: np.ndarray, fs: float = 1.0) -> None:
    """Plot clean vs noisy signal in time domain and their power spectra."""
    t = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle("Additive White Gaussian Noise (AWGN)")

    # Time domain — clean
    axes[0, 0].plot(t, signal, color="steelblue")
    axes[0, 0].set_title("Clean Signal")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude")

    # Time domain — noisy
    axes[0, 1].plot(t, noisy_signal, color="tomato", alpha=0.8)
    axes[0, 1].set_title(f"Noisy Signal  (SNR = {compute_snr_db(signal, noisy_signal):.1f} dB)")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Amplitude")

    # Power spectrum — clean
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    psd_clean = np.abs(np.fft.rfft(signal)) ** 2
    axes[1, 0].semilogy(freqs, psd_clean, color="steelblue")
    axes[1, 0].set_title("Power Spectrum — Clean")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Power")

    # Power spectrum — noisy (white noise floor is visible)
    psd_noisy = np.abs(np.fft.rfft(noisy_signal)) ** 2
    axes[1, 1].semilogy(freqs, psd_noisy, color="tomato", alpha=0.8)
    axes[1, 1].set_title("Power Spectrum — Noisy")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_ylabel("Power")

    plt.tight_layout()
    plt.savefig("awgn_plot.png", dpi=150)
    plt.show()
    print("Plot saved to awgn_plot.png")


def demo() -> None:
    """Demonstrate AWGN on a multi-tone sinusoidal signal at several SNR levels."""
    rng = np.random.default_rng(42)
    np.random.seed(42)

    fs = 1000.0          # sampling frequency (Hz)
    duration = 1.0       # seconds
    t = np.arange(int(fs * duration)) / fs

    # Clean signal: sum of three sinusoids
    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
        + 0.3 * np.sin(2 * np.pi * 200 * t)
    )

    print("SNR sweep demo")
    print("-" * 30)
    for snr_target in [20, 10, 5, 0]:
        noisy = awgn(signal, snr_db=snr_target)
        actual_snr = compute_snr_db(signal, noisy)
        print(f"Target SNR: {snr_target:4d} dB  |  Actual SNR: {actual_snr:.2f} dB")

    # Plot at 10 dB SNR
    noisy_10db = awgn(signal, snr_db=10)
    plot_awgn(signal, noisy_10db, fs=fs)


if __name__ == "__main__":
    demo()
