"""
feature_extraction.py
---------------------
Handles audio loading, log-mel spectrogram computation, and
frame-level feature extraction for the DCASE 2023 Task 2 baseline.

Audio files are 10-second mono clips at 16 kHz.
We extract 128-bin log-mel spectrograms and concatenate 5 adjacent
frames to form 640-dimensional input vectors, reproducing the setup
described in the baseline system (Dohi et al., 2023).
"""

import os
import numpy as np
import librosa


def load_audio(file_path, sr=16000, duration=10):
    """
    Load an audio file and ensure it has correct length.

    Parameters
    ----------
    file_path : str
        Path to the .wav file.
    sr : int
        Target sampling rate.
    duration : int
        Expected clip duration in seconds.

    Returns
    -------
    y : np.ndarray
        Audio time-series of shape (sr * duration,).
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    target_len = sr * duration
    if len(y) < target_len:
        # zero-pad short clips
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y


def extract_log_mel(y, sr=16000, n_fft=1024, hop_length=512,
                    n_mels=128, power=2.0, fmin=0, fmax=8000):
    """
    Compute the log-mel spectrogram for an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series.
    sr, n_fft, hop_length, n_mels, power, fmin, fmax :
        Standard librosa mel-spectrogram parameters.

    Returns
    -------
    log_mel : np.ndarray
        Log-mel spectrogram of shape (n_mels, T) where T depends on
        the audio length, n_fft, and hop_length.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=power, fmin=fmin, fmax=fmax
    )
    log_mel = 20.0 * np.log10(np.maximum(mel_spec, 1e-10))
    return log_mel


def create_frame_vectors(log_mel, n_frames=5):
    """
    Slide a window of `n_frames` adjacent frames across the spectrogram
    and concatenate them to form input vectors.

    For a 128-mel-bin spectrogram with n_frames=5, each vector has
    dimension 128 * 5 = 640.

    Parameters
    ----------
    log_mel : np.ndarray
        Log-mel spectrogram of shape (n_mels, T).
    n_frames : int
        Number of consecutive frames to concatenate.

    Returns
    -------
    vectors : np.ndarray
        Shape (T - n_frames + 1, n_mels * n_frames).
    """
    n_mels, total_frames = log_mel.shape
    if total_frames < n_frames:
        raise ValueError(
            f"Spectrogram has {total_frames} frames but need at least "
            f"{n_frames} frames."
        )

    vectors = []
    for t in range(total_frames - n_frames + 1):
        frame_block = log_mel[:, t:t + n_frames]      # (n_mels, n_frames)
        vectors.append(frame_block.flatten())          # (n_mels * n_frames,)

    return np.array(vectors, dtype=np.float32)


def extract_features_from_file(file_path, sr=16000, n_fft=1024,
                                hop_length=512, n_mels=128,
                                n_frames=5, duration=10):
    """
    End-to-end feature extraction: load audio -> log-mel -> frame vectors.

    Parameters
    ----------
    file_path : str
        Path to .wav file.

    Returns
    -------
    vectors : np.ndarray
        Shape (num_vectors, n_mels * n_frames).
    """
    y = load_audio(file_path, sr=sr, duration=duration)
    log_mel = extract_log_mel(y, sr=sr, n_fft=n_fft,
                               hop_length=hop_length, n_mels=n_mels)
    vectors = create_frame_vectors(log_mel, n_frames=n_frames)
    return vectors


def extract_features_from_directory(directory, sr=16000, n_fft=1024,
                                     hop_length=512, n_mels=128,
                                     n_frames=5, duration=10):
    """
    Extract features from all .wav files in a directory.

    Parameters
    ----------
    directory : str
        Folder containing .wav files.

    Returns
    -------
    all_vectors : np.ndarray
        Concatenated feature vectors from all files.
    file_lengths : list of int
        Number of vectors per file (needed to map scores back to files).
    file_list : list of str
        Ordered list of processed file names.
    """
    all_vectors = []
    file_lengths = []
    file_list = []

    wav_files = sorted([
        f for f in os.listdir(directory) if f.endswith(".wav")
    ])

    for fname in wav_files:
        fpath = os.path.join(directory, fname)
        vectors = extract_features_from_file(
            fpath, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, n_frames=n_frames, duration=duration
        )
        all_vectors.append(vectors)
        file_lengths.append(len(vectors))
        file_list.append(fname)

    if all_vectors:
        all_vectors = np.concatenate(all_vectors, axis=0)
    else:
        all_vectors = np.array([], dtype=np.float32)

    return all_vectors, file_lengths, file_list
