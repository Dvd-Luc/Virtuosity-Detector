
def tf_to_yolo(t0, t1, f0, f1, t_max, f_max):
    """
    Convert time-frequency bounding box coordinates to YOLO format.

    The function maps a bounding box defined by start/end times (t0, t1)
    and frequency bounds (f0, f1) to normalized YOLO coordinates
    (x_center, y_center, width, height), where x and y are in [0,1] relative
    to the total duration and frequency range.

    Parameters
    ----------
    t0 : float
        Start time of the bounding box (seconds).
    t1 : float
        End time of the bounding box (seconds).
    f0 : float
        Minimum frequency of the bounding box (Hz).
    f1 : float
        Maximum frequency of the bounding box (Hz).
    t_max : float
        Maximum time of the spectrogram (seconds).
    f_max : float
        Maximum frequency of the spectrogram (Hz).

    Returns
    -------
    xc : float
        Normalized x-coordinate of the bounding box center.
    yc : float
        Normalized y-coordinate of the bounding box center (inverted, 0=bottom).
    w : float
        Normalized width of the bounding box.
    h : float
        Normalized height of the bounding box.
    """

    xc = ((t0 + t1) / 2) / t_max
    w  = (t1 - t0) / t_max
    yc = 1 - ((f0 + f1) / 2) / f_max
    h  = (f1 - f0) / f_max
    return xc, yc, w, h

# def save_temp_spectro(y, sr, out_path, n_fft=512, hop_length=128):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
#     S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

#     plt.figure(figsize=(5,5))
#     plt.imshow(S_db, origin="lower", aspect="auto", cmap="gray", extent=[0, len(y)/sr, 0, sr/2])
#     plt.axis("off")
#     plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
#     plt.close()

def yolo_to_tf_centered(xc, yc, w, h, win_len, f_max):
    """
    Convert YOLO normalized bounding box coordinates to absolute
    time–frequency bounds within a centered spectrogram window.

    This function assumes a YOLO format where:
    - time corresponds to the x-axis
    - frequency corresponds to the y-axis (inverted, top = high frequency)
    - all coordinates are normalized in the [0, 1] range

    Parameters:
        xc (float): Normalized x-coordinate of the bounding box center (time axis).
        yc (float): Normalized y-coordinate of the bounding box center (frequency axis, inverted).
        w (float): Normalized width of the bounding box (relative to window duration).
        h (float): Normalized height of the bounding box (relative to frequency range).
        win_len (float): Duration of the spectrogram window in seconds.
        f_max (float): Maximum frequency represented in the spectrogram (Hz).

    Returns:
        tuple:
            - t_start (float): Start time of the bounding box (seconds, relative to window).
            - t_end (float): End time of the bounding box (seconds, relative to window).
            - f_min (float): Minimum frequency of the bounding box (Hz).
            - f_max_val (float): Maximum frequency of the bounding box (Hz).

    Notes:
        - The frequency axis is inverted to match YOLO image coordinates
          (y = 0 at the top of the image).
        - Returned time values are relative to the centered window, not absolute
          positions in the original audio signal.
    """

    t_start = (xc - w/2) * win_len
    t_end   = (xc + w/2) * win_len
    # Flips the y-axis to convert from YOLO image coordinates to frequency values
    yc_real = 1 - yc
    f_min = (yc_real - h/2) * f_max
    f_max_val = (yc_real + h/2) * f_max
    return t_start, t_end, f_min, f_max_val

def yolo_to_tf_xyxy(x1, x2, y1, y2, win_len, f_max):
    """
    Convert YOLO normalized bounding box coordinates (x1, x2, y1, y2)
    to absolute time–frequency bounds.

    This function assumes YOLO predictions are expressed as normalized
    corner coordinates in the image space:
    - x1, x2 correspond to the time axis
    - y1, y2 correspond to the frequency axis (inverted, top = high frequency)

    Parameters:
        x1 (float): Normalized left x-coordinate of the bounding box.
        x2 (float): Normalized right x-coordinate of the bounding box.
        y1 (float): Normalized top y-coordinate of the bounding box.
        y2 (float): Normalized bottom y-coordinate of the bounding box.
        win_len (float): Duration of the spectrogram window in seconds.
        f_max (float): Maximum frequency represented in the spectrogram (Hz).

    Returns:
        tuple:
            - t_start (float): Start time of the bounding box (seconds, relative to window).
            - t_end (float): End time of the bounding box (seconds, relative to window).
            - f_min (float): Minimum frequency of the bounding box (Hz).
            - f_max_val (float): Maximum frequency of the bounding box (Hz).

    Notes:
        - The frequency axis is inverted to match YOLO image coordinates
          (y = 0 corresponds to the top of the spectrogram).
        - Time bounds are relative to the centered spectrogram window.
    """
    
    t_start = x1 * win_len
    t_end = x2 * win_len
    yc1_real = 1 - y1
    yc2_real = 1 - y2
    f_min = min(yc1_real, yc2_real) * f_max
    f_max_val = max(yc1_real, yc2_real) * f_max
    return t_start, t_end, f_min, f_max_val

def load_yolo_prediction(txt_path, win_len, f_max):
    """
    Load a YOLO prediction from a label file and convert it to
    time–frequency coordinates.

    The function reads a YOLO `.txt` prediction file containing
    normalized bounding box parameters and converts them into
    absolute time and frequency bounds relative to the spectrogram
    window.

    Parameters:
        txt_path (Path): Path to the YOLO prediction `.txt` file.
        win_len (float): Duration of the spectrogram window in seconds.
        f_max (float): Maximum frequency represented in the spectrogram (Hz).

    Returns:
        dict or None:
            A dictionary containing the converted prediction:
                - 't_start_rel' (float): Start time relative to the window (seconds).
                - 't_end_rel' (float): End time relative to the window (seconds).
                - 'f_min' (float): Minimum frequency of the detected region (Hz).
                - 'f_max' (float): Maximum frequency of the detected region (Hz).
                - 'confidence' (float or None): YOLO confidence score, if available.
            Returns None if the file does not exist, is empty, or contains
            an invalid prediction.

    Notes:
        - Assumes YOLO format: `class xc yc w h [confidence]`.
        - Only the first prediction in the file is loaded.
        - Coordinates are converted using `yolo_to_tf_centered`.
    """
    
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return None
    
    with open(txt_path) as f:
        line = f.readline().strip()
    
    parts = line.split()
    if len(parts) < 5:
        return None
    
    _, xc, yc, w, h = map(float, parts[:5])
    confidence = float(parts[5]) if len(parts) > 5 else None
    
    t_start_rel, t_end_rel, f_min, f_max_val = yolo_to_tf_centered(xc, yc, w, h, win_len, f_max)
    
    return {
        't_start_rel': t_start_rel,
        't_end_rel': t_end_rel,
        'f_min': f_min,
        'f_max': f_max_val,
        'confidence': confidence
    }