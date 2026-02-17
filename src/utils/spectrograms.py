import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_centered_segment(y, sr, start, end, win_len):
    """
    Extract a fixed-length audio segment centered between two time markers.

    The function computes the midpoint between `start` and `end` (in seconds),
    then extracts a segment of duration `win_len` centered on this midpoint.
    If the segment exceeds the signal boundaries, it is shifted to fit within
    the valid signal duration.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sampling rate (Hz).
    start : float
        Start time of the target region (seconds).
    end : float
        End time of the target region (seconds).
    win_len : float
        Desired segment length (seconds).

    Returns
    -------
    y_segment : np.ndarray
        Extracted audio segment.
    seg_start : float
        Actual start time of the extracted segment (seconds).
    seg_end : float
        Actual end time of the extracted segment (seconds).
    """
    sig_duration = len(y) / sr
    center = 0.5 * (start + end)
    seg_start = max(0, center - win_len / 2)
    seg_end = seg_start + win_len
    seg_end = min(sig_duration, seg_end)

    i0 = int(seg_start * sr)
    i1 = int(seg_end * sr)
    return y[i0:i1], seg_start, seg_end


def save_spectrogram(y, sr, out_png, cfg):
    """
    Compute and save a spectrogram of an audio signal as an image.

    The function performs a Short-Time Fourier Transform (STFT) on the
    input signal, converts the amplitude to decibels, and saves the
    spectrogram as a PNG image with the specified colormap and figure settings.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D array).
    sr : int
        Sampling rate of the audio signal (Hz).
    out_png : str
        Path where the spectrogram image will be saved.
    cfg : dict
        Configuration dictionary with keys:
        - 'n_fft' : int, FFT window size
        - 'hop_length' : int, hop length for STFT
        - 'cmap' : str or matplotlib colormap, color map for the spectrogram
        - 'win_len' : float, duration of the window in seconds for plotting
        - 'sr' : int, sampling rate used to scale frequency axis

    Returns
    -------
    None
        The spectrogram is saved to the specified file path.
    """

    S = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    S = np.abs(S)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(5,5))
    plt.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        cmap=cfg.cmap,
        extent=[0, cfg.win_len, 0, cfg.sr/2]
    )
    plt.axis("off")
    plt.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close()