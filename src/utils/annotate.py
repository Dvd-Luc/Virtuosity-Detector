import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("TkAgg")  # ou "Qt5Agg"
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, TextBox, Button
from matplotlib.patches import Rectangle
import sounddevice as sd
from src.config import TrillConfig


def annotate(signal, sr, config:TrillConfig, prediction = None):
    """
    Display a spectrogram of an audio segment and allow the user to manually annotate trill regions.

    Parameters
    ----------
    signal : np.ndarray
        The audio waveform of the segment to annotate.
    sr : int
        The sampling rate of the audio signal.
    config : TrillConfig
        Configuration object containing parameters for STFT, plotting, etc.
    prediction : dict, optional
        Optional pre-filled prediction with keys:
            - "t_min": float, start time of predicted trill
            - "t_max": float, end time of predicted trill
            - "f_min": float, minimum frequency of predicted trill
            - "f_max": float, maximum frequency of predicted trill
            - "count": int, predicted number of trills
        If provided, the spectrogram will display the predicted region.

    Returns
    -------
    selected : dict
        Dictionary containing the final manual annotation:
            - "t_min": float, start time of the annotated trill
            - "t_max": float, end time of the annotated trill
            - "f_min": float, minimum frequency of the annotated trill
            - "f_max": float, maximum frequency of the annotated trill
            - "count": int, number of trills annotated
            - "no_trill": bool, True if marked as containing no trill

    Notes
    -----
    - Opens an interactive matplotlib figure with the spectrogram.
    - Allows drawing a rectangle to select the trill region.
    - Provides a textbox to enter the number of trills.
    - Includes buttons:
        - "No Trill" to mark the segment as having no trills.
        - "Play" to listen to the audio segment.
        - "Validate" to confirm the annotation.
    - If `prediction` is provided, it is displayed as a dashed orange rectangle and pre-fills the count.
    """
    
    S = np.abs(librosa.stft(signal, n_fft=config.n_fft, hop_length=config.hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    fig.suptitle("Select trill region and enter count", fontsize=16)


    librosa.display.specshow(
        S_db, sr=sr, hop_length=config.hop_length,
        x_axis="time", y_axis="log", ax=ax
    )

    selected = {
        "t_min": None,
        "t_max": None,
        "f_min": None,
        "f_max": None,
        "count": None,
        "no_trill": False
    }
    rect_patch = {"patch": None}


    if prediction is not None:
        print(f"Pre-filled prediction : T_min={prediction['t_min']:.2f}s, T_max={prediction['t_max']:.2f}s, F_min={prediction['f_min']:.1f}Hz, F_max={prediction['f_max']:.1f}Hz, Count={prediction['count']}")
        t_min = prediction["t_min"]
        t_max = prediction["t_max"]
        f_min = prediction["f_min"]
        f_max = prediction["f_max"]
        count = prediction["count"]
        rect_patch["patch"] = ax.add_patch(
            Rectangle(
                (t_min, f_min), 
                t_max - t_min,
                f_max - f_min, 
                fill=False,
                edgecolor="orange",
                linewidth=1,
                linestyle="--"
            )
        )

        selected["t_min"] = t_min
        selected["t_max"] = t_max
        selected["f_min"] = f_min
        selected["f_max"] = f_max
        selected["count"] = count

    def onselect(eclick, erelease):
        t_min, t_max = sorted([eclick.xdata, erelease.xdata])
        f_min, f_max = sorted([eclick.ydata, erelease.ydata])

        selected["t_min"] = t_min
        selected["t_max"] = t_max
        selected["f_min"] = f_min
        selected["f_max"] = f_max
        selected["no_trill"] = False

        if rect_patch["patch"]:
            rect_patch["patch"].remove()

        rect_patch["patch"] = ax.add_patch(
            plt.Rectangle(
                (t_min, f_min),
                t_max - t_min,
                f_max - f_min,
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
        )

    fig.rectangle = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        interactive=True
    )


    axbox = plt.axes([0.1, 0.1, 0.10, 0.05])
    textbox = TextBox(axbox, "Trills", initial=str(selected["count"] or ""))

    def submit(text):
        try:
            selected["count"] = int(text)
        except:
            pass

    textbox.on_submit(submit)

    # ===== Bouton "No Trill" =====
    ax_notrill = plt.axes([0.25, 0.1, 0.12, 0.05])
    notrill_button = Button(ax_notrill, "No Trill", color='lightcoral')

    def mark_no_trill(event, timeout_sec=2):
        """Marque l'échantillon comme ne contenant pas de trill"""
        selected["no_trill"] = True
        selected["count"] = 0
        # Coordonnées nulles = masque vide
        selected["t_min"] = 0.0
        selected["t_max"] = 0.0
        selected["f_min"] = 0.0
        selected["f_max"] = 0.0
        
        # Retirer le rectangle si présent
        if rect_patch["patch"]:
            rect_patch["patch"].remove()

        textbox.set_val("0")

        no_trill_text = ax.text(0.5, 0.5, "NO TRILL", 
                transform=ax.transAxes,
                fontsize=40, color='red', alpha=0.5,
                ha='center', va='center',
                weight='bold')
        fig.canvas.draw_idle()
        timer = fig.canvas.new_timer(interval=int(timeout_sec * 1000))
        
        def clear_text():
            no_trill_text.remove()
            fig.canvas.draw_idle()


        timer.add_callback(clear_text)
        timer.start()

    notrill_button.on_clicked(mark_no_trill)

    axplay = plt.axes([0.5, 0.1, 0.1, 0.05])
    play_button = Button(axplay, "▶ Play")
    def play_audio(event):
        sd.stop()
        sd.play(signal, sr)
    play_button.on_clicked(play_audio)

    #===== "Validate" Button =====
    axval = plt.axes([0.7, 0.05, 0.2, 0.08])
    button = Button(axval, "Validate")

    def validate(event, timeout=2):
        if (
            selected["t_min"] is None
            or selected["t_max"] is None
            or selected["count"] is None
        ):
            warning_text = ax.text(
                0.5, 0.95, "⚠️ Incomplete annotation!",
                transform=ax.transAxes,
                fontsize=12,
                color="red",
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8)
                )
            fig.canvas.draw_idle()
            timer = fig.canvas.new_timer(interval=int(timeout * 1000))
            
            def remove_warning():
                warning_text.remove()
                fig.canvas.draw_idle()
            
            timer.add_callback(remove_warning)
            timer.start()

            print("Incomplete annotation.")
            return
        plt.close(fig)

    button.on_clicked(validate)

    plt.show()

    return selected