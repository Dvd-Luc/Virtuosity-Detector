import librosa
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt


def band_filter(y, sr, filter_type='highpass', low_freq=None, high_freq=None, order=4):
    """
    Apply a Butterworth filter to an audio signal.
    
    Supports high-pass, low-pass, and band-pass filtering with configurable
    cutoff frequencies and filter order. Uses zero-phase filtering (filtfilt)
    to avoid phase distortion.
    
    Parameters
    ----------
    y : np.ndarray
        Input audio signal.
    sr : int
        Sampling rate of the audio signal (Hz).
    filter_type : {'highpass', 'lowpass', 'bandpass'}, default='highpass'
        Type of filter to apply:
        - 'highpass': Attenuates frequencies below `low_freq`
        - 'lowpass': Attenuates frequencies above `high_freq`
        - 'bandpass': Keeps frequencies between `low_freq` and `high_freq`
    low_freq : float, optional
        Lower cutoff frequency (Hz). Required for 'highpass' and 'bandpass'.
    high_freq : float, optional
        Upper cutoff frequency (Hz). Required for 'lowpass' and 'bandpass'.
    order : int, default=4
        Filter order. Higher order = steeper roll-off.
        Common values:
        - order=4: 24 dB/octave attenuation (Podos paper)
        - order=3: 18 dB/octave
        - order=8: 48 dB/octave (very steep)
    
    Returns
    -------
    y_filtered : np.ndarray
        Filtered audio signal (same shape as input).
    
    Raises
    ------
    ValueError
        If required cutoff frequencies are not provided for the chosen filter type.
    
    Notes
    -----
    - Uses a Butterworth filter (maximally flat passband).
    - Zero-phase filtering (filtfilt) is applied to avoid time-domain distortion.
    - The effective attenuation is order × 6 dB/octave (e.g., order=4 → 24 dB/octave).
    - For filter_type='bandpass', both low_freq and high_freq must be provided.
    - Cutoff frequencies are normalized to Nyquist frequency (sr / 2).
    """

    nyquist = sr / 2.0
    
    if filter_type == 'highpass':
        if low_freq is None:
            raise ValueError("low_freq must be provided for highpass filter")
        if low_freq >= nyquist:
            raise ValueError(f"low_freq ({low_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
        
        Wn = low_freq / nyquist
        btype = 'high'
    
    elif filter_type == 'lowpass':
        if high_freq is None:
            raise ValueError("high_freq must be provided for lowpass filter")
        if high_freq >= nyquist:
            raise ValueError(f"high_freq ({high_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
        
        Wn = high_freq / nyquist
        btype = 'low'
    
    elif filter_type == 'bandpass':
        if low_freq is None or high_freq is None:
            raise ValueError("Both low_freq and high_freq must be provided for bandpass filter")
        if low_freq >= high_freq:
            raise ValueError(f"low_freq ({low_freq}) must be less than high_freq ({high_freq})")
        if high_freq >= nyquist:
            raise ValueError(f"high_freq ({high_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
        
        Wn = [low_freq / nyquist, high_freq / nyquist]
        btype = 'band'
    
    else:
        raise ValueError(f"Invalid filter_type '{filter_type}'. Must be 'highpass', 'lowpass', or 'bandpass'")
    
    # Use sos (second-order sections) for numerical stability with high orders
    sos = butter(order, Wn, btype=btype, output='sos')
    
    # Apply zero-phase filtering
    y_filtered = sosfiltfilt(sos, y)
    
    return y_filtered

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

def get_bandwidth_podos(y, sr, n_fft, hop_length):
    """
    Calculate the frequency bandwidth of a trill using the Podos method.
    
    The bandwidth is defined as the difference between the minimum and maximum
    frequencies at -24 dB relative to the peak amplitude across the entire
    spectrogram.
    
    Parameters
    ----------
    y : np.ndarray
        Audio signal containing the trill.
    sr : int
        Sampling rate of the audio signal (Hz).
    n_fft : int
        FFT window size for computing the spectrogram.
    hop_length : int
        Number of samples between successive frames in the STFT.
    
    Returns
    -------
    f_min : float
        Minimum frequency at -24 dB threshold (Hz).
    f_max : float
        Maximum frequency at -24 dB threshold (Hz).
    
    Notes
    -----
    - Computes the STFT and converts to dB scale.
    - Finds the global peak amplitude across the entire spectrogram.
    - Determines all frequencies where amplitude >= (peak - 24 dB).
    - Returns the minimum and maximum of those frequencies.
    
    References
    ----------
    Podos, J. (1997). A performance constraint on the evolution of trilled
    vocalizations in a songbird family (Passeriformes: Emberizidae).
    Evolution, 51(2), 537-551.

    """
    y_filtered = band_filter(y, sr, filter_type='highpass', low_freq=1000, order=4)
    
    S = np.abs(librosa.stft(y_filtered, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    peak_db = np.max(S_db)
    
    threshold_db = peak_db - 24.0
    
    # across all time frames
    above_threshold = np.any(S_db >= threshold_db, axis=1)
    
    freq_indices = np.where(above_threshold)[0]
    
    if len(freq_indices) == 0:
        return 0.0, 0.0
    
    f_min = freqs[freq_indices[0]]
    f_max = freqs[freq_indices[-1]]
    
    return f_min, f_max
