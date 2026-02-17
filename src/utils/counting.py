import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks

def highpass_env(env, fs, cutoff=10):
    """
    Apply a high-pass Butterworth filter to an envelope signal.

    Parameters
    ----------
    env : np.ndarray
        The input envelope signal to be filtered.
    fs : float
        The sampling rate of the signal in Hz.
    cutoff : float, optional
        The cutoff frequency of the high-pass filter in Hz (default is 10 Hz).

    Returns
    -------
    np.ndarray
        The filtered envelope signal with frequencies below the cutoff removed.

    Notes
    -----
    - Uses a 3rd-order Butterworth filter.
    - Filtering is applied with zero-phase distortion using `filtfilt`.
    - Useful for removing slow trends or baseline drift in amplitude envelopes.
    """

    b, a = butter(3, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, env)

def correct_subharmonic(freqs, spectrum, f0):
    """
    Correct the fundamental frequency estimate if a strong subharmonic is present.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequency bins corresponding to the spectrum.
    spectrum : np.ndarray
        Magnitude spectrum of the signal.
    f0 : float
        Initial fundamental frequency estimate. This is the frequency we want to check for potential subharmonic issues.
    Returns
    -------
    float
        Corrected fundamental frequency. If a strong subharmonic (2×f0) is detected, returns 2×f0. Otherwise, returns the original f0.
    """ 
    
    idx_f = np.argmin(np.abs(freqs - f0))
    idx_2f = np.argmin(np.abs(freqs - 2*f0))

    if idx_2f < len(spectrum):
        if spectrum[idx_2f] > 0.5 * spectrum[idx_f]:
            return 2 * f0
    return f0



def trill_rate_detection_am2(
    signal,
    sample_rate=48000,
    trill_min_freq=1000,
    trill_max_freq=9000,
    n_fft=256,
    hop_length=64,
    min_rate=4,
    max_rate=200,
):
    """
    Estimate the trill rate of an audio signal using amplitude modulation analysis.

    Parameters
    ----------
    signal : np.ndarray
        Audio waveform containing the trill to analyze.
    sample_rate : float, optional
        Sampling rate of the audio signal in Hz (default is 48000 Hz).
    trill_min_freq : float, optional
        Minimum frequency of the spectral band to analyze (Hz, default 1000 Hz).
    trill_max_freq : float, optional
        Maximum frequency of the spectral band to analyze (Hz, default 9000 Hz).
    n_fft : int, optional
        FFT size for computing the short-time Fourier transform (default 256).
    hop_length : int, optional
        Hop length for STFT (default 64 samples).
    min_rate : float, optional
        Minimum trill rate to consider in Hz (default 4 Hz).
    max_rate : float, optional
        Maximum trill rate to consider in Hz (default 200 Hz).

    Returns
    -------
    trill_rate : float
        Estimated trill rate in Hz (frequency of amplitude modulation peak within the band).
        Returns 0.0 if no valid modulation frequency is found in the specified range.
    env : np.ndarray
        Processed amplitude envelope of the spectral band used for rate estimation.
    freqs_env : np.ndarray
        Frequency axis corresponding to `env` after envelope FFT (Hz).
    fft_env : np.ndarray
        Magnitude spectrum of the envelope used to identify the trill rate.

    Notes
    -----
    - The function computes the STFT of the signal, extracts the specified frequency band,
      computes a robust amplitude envelope using the 90th percentile, and applies a log transform.
    - A high-pass Butterworth filter (cutoff 10 Hz) is applied to remove slow trends in the envelope.
    - The FFT of the envelope is used to identify the dominant modulation frequency corresponding
      to the trill rate.
    - Useful for analyzing bird song trills or other rapid amplitude modulations in audio signals.
    """

    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band = (freqs >= trill_min_freq) & (freqs <= trill_max_freq)
    env = np.percentile(S[band], 90, axis=0)
    env = np.log(env + 1e-8)
    env -= env.mean()
    fs_env = sample_rate / hop_length
    env = highpass_env(env, fs_env, cutoff=min_rate/2)
    n_fft_env = 4 * int(2 ** np.ceil(np.log2(len(env))))
    fft_env = np.abs(np.fft.rfft(env, n=n_fft_env))
    freqs_env = np.fft.rfftfreq(n_fft_env, d=1/fs_env)
    valid = (freqs_env >= min_rate) & (freqs_env <= max_rate)
    if not np.any(valid):
        return 0.0, env, freqs_env, fft_env
    idx = np.argmax(fft_env[valid])
    trill_rate = freqs_env[valid][idx]
    return trill_rate, env, freqs_env, fft_env

def trill_rate_robust_fixed(env, fs_env, rate_fft, min_rate=4, max_rate=200, debug=False):
    """
    Robust trill rate estimation using both FFT and autocorrelation peaks.

    This function takes an amplitude envelope of a signal and its sampling frequency,
    then estimates the trill rate by combining information from:
        - The initial FFT-based estimate (rate_fft)
        - Peaks in the autocorrelation of the envelope
        - Harmonic consistency and peak scoring

    Steps:
        1. Normalize the envelope and compute its autocorrelation.
        2. Identify candidate trill rates from FFT and autocorrelation peaks, including octaves (0.5×, 1×, 2×).
        3. Deduplicate candidates that are within 5% of each other.
        4. Score candidates based on:
            - Autocorrelation amplitude at fundamental lag
            - Alignment with detected autocorrelation peaks
            - Presence of harmonics (2×, 3×)
            - Proximity to the FFT-based rate
            - Penalization for low subharmonics
        5. Return the best candidate trill rate along with intermediate data for debugging or plotting.

    Parameters:
        env (np.ndarray): Amplitude envelope of the signal.
        fs_env (float): Sampling frequency of the envelope (Hz).
        rate_fft (float): Initial trill rate estimated from FFT (Hz).
        min_rate (float, optional): Minimum plausible trill rate (Hz). Default is 4 Hz.
        max_rate (float, optional): Maximum plausible trill rate (Hz). Default is 200 Hz.
        debug (bool, optional): If True, prints intermediate candidates, scores, and selection. Default is False.

    Returns:
        best_rate (float): Estimated trill rate in Hz. Falls back to rate_fft if no robust candidate found.
        ac (np.ndarray): Autocorrelation of the envelope.
        lags (np.ndarray): Time lags corresponding to the autocorrelation.
        peak_lags (np.ndarray or list): Lags of the top autocorrelation peaks considered.
    """

    # Autocorrelation
    env = env - np.mean(env)
    ac = np.correlate(env, env, mode="full")
    ac = ac[len(ac)//2:]
    ac /= np.max(ac) + 1e-8
    lags = np.arange(len(ac)) / fs_env
    
    min_lag = 1 / max_rate
    max_lag = 1 / min_rate
    valid = (lags >= min_lag) & (lags <= max_lag)
    
    if not np.any(valid):
        return rate_fft, ac, lags, []
    
    peaks, props = find_peaks(
        ac[valid], 
        prominence=0.1,
        distance=int(fs_env*0.005)
    )
    
    if len(peaks) == 0:
        return rate_fft, ac, lags, []
    
    if debug:
        print(f"\n=== Trill Rate Robust Debug ===")
        print(f"Rate FFT: {rate_fft:.2f} Hz")
        print(f"Pics trouvés: {len(peaks)}")
    
    # Generate candidate rates from both FFT and autocorrelation peaks, including octaves
    candidates_info = []
    
    # 1. FFT initial candidate and its octaves
    for factor in [0.5, 1.0, 2.0]:
        candidate = rate_fft * factor
        if min_rate <= candidate <= max_rate:
            candidates_info.append({
                'rate': candidate,
                'source': f'FFT×{factor}',
                'factor': factor
            })
    
    # 2. Candidates from autocorrelation peaks and their octaves
    for i, peak in enumerate(peaks[:8]):  # Up to 8 peaks
        peak_lag = lags[valid][peak]
        peak_rate = 1 / peak_lag
        peak_amplitude = ac[valid][peak]
        
        for factor in [0.5, 1.0, 2.0]:
            candidate = peak_rate * factor
            if min_rate <= candidate <= max_rate:
                candidates_info.append({
                    'rate': candidate,
                    'source': f'AC_peak{i+1}×{factor}',
                    'peak_amplitude': peak_amplitude,
                    'peak_index': i,
                    'factor': factor
                })
    
    # Deduplicate candidates that are within 5% of each other
    unique_candidates = []
    for cand in candidates_info:
        is_duplicate = False
        for existing in unique_candidates:
            if abs(cand['rate'] - existing['rate']) / cand['rate'] < 0.05:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(cand)
    
    if debug:
        print(f"Unique candiates: {len(unique_candidates)}")
    
    # Improved scoring
    best_candidate = None
    best_score = -np.inf
    
    for cand in unique_candidates:
        candidate_rate = cand['rate']
        score = 0
        
        # CRIT 1. Autocorrelation amplitude at fundamental lag
        fundamental_lag = 1 / candidate_rate
        if min_lag <= fundamental_lag <= max_lag:
            idx = np.argmin(np.abs(lags[valid] - fundamental_lag))
            fundamental_ac = ac[valid][idx]
            score += fundamental_ac * 10.0  # Poids fort
            
            if debug and fundamental_ac > 0.5:
                print(f"  {candidate_rate:6.2f} Hz: fundamental_ac={fundamental_ac:.3f}")
        
        # CRIT 2. Alignment with detected peaks
        # Bonus if one of the top peaks is close to the candidate rate
        for peak in peaks[:8]:
            peak_lag = lags[valid][peak]
            peak_rate = 1 / peak_lag
            if abs(peak_rate - candidate_rate) / candidate_rate < 0.05:
                score += 3.0
                break
        
        # CRIT 3. Harmonic consistency
        # Verify if there are peaks at 2× and 3× the candidate rate
        harmonic_support = 0
        for h in [2, 3]:
            harmonic_lag = h / candidate_rate
            if min_lag <= harmonic_lag <= max_lag:
                idx_h = np.argmin(np.abs(lags[valid] - harmonic_lag))
                harmonic_ac = ac[valid][idx_h]
                harmonic_support += harmonic_ac
        
        score += harmonic_support * 1.0
        
        # CRIT 4. Proximity to FFT-based rate
        # Bonus if the candidate is close to the initial FFT estimate
        fft_diff = abs(candidate_rate - rate_fft) / rate_fft
        if fft_diff < 0.15:
            score += 2.0 * (1 - fft_diff)
        
        # CRIT 5. Penalize low subharmonics (e.g., 0.5×) if they are not supported by strong autocorrelation
        # This helps avoid selecting a subharmonic that is not actually present in the signal

        if abs(candidate_rate - rate_fft/2) / rate_fft < 0.1:
            score -= 1.0 
        
        if debug:
            print(f"  Candidate {candidate_rate:6.2f} Hz ({cand['source']:12s}): score={score:.2f}")
        
        if score > best_score:
            best_score = score
            best_candidate = cand
    
    if debug and best_candidate:
        print(f"→ Chosen: {best_candidate['rate']:.2f} Hz (score={best_score:.2f})")
    
    peak_lags = lags[valid][peaks[:8]] if len(peaks) > 0 else None

    if best_candidate:
        return best_candidate['rate'], ac, lags, peak_lags
    else:
        return rate_fft, ac, lags, peak_lags

def estimate_trill_rate(signal, sample_rate=48000, hop_length=64, debug=False):
    """
    Estimate the trill rate of an audio signal using a two-stage approach.

    This function first estimates the trill rate using a frequency-domain
    analysis of the amplitude envelope (FFT-based method), then refines
    this estimate using a robust autocorrelation-based method.

    Workflow:
        1. Compute an initial trill rate estimate using `trill_rate_detection_am2`,
           based on spectral energy modulation.
        2. Convert the STFT hop length into an envelope sampling rate.
        3. Refine the estimate using `trill_rate_robust_fixed`, which combines
           autocorrelation peaks, harmonic consistency, and the FFT estimate.

    Parameters:
        signal (np.ndarray): Audio signal containing a trill.
        sample_rate (int, optional): Sampling rate of the audio signal (Hz).
            Default is 48000.
        hop_length (int, optional): Hop length used for the STFT, needed to
            compute the envelope sampling frequency. Default is 64.
        debug (bool, optional): If True, enables verbose output and debugging
            information from the robust estimator. Default is False.

    Returns:
        float: Estimated trill rate in Hz.
    """
    rate_fft, env, freqs_env, fft_env = trill_rate_detection_am2(signal, sample_rate)
    fs_env = sample_rate / hop_length  # hop_length utilisé dans stft
    rate_robust, ac, lags, peak_lags = trill_rate_robust_fixed(env, fs_env, rate_fft, debug=debug)
    return rate_robust