import librosa
import librosa.display
import json
import numpy as np
from fastdtw import fastdtw
import matplotlib.pyplot as plt


def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr)
    return y, sr


def extract_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T


def load_phoneme_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_mfcc_to_phonemes(mfcc, phonemes):
    step = mfcc.shape[0] // len(phonemes)
    target_seq = [np.mean(mfcc[i*step:(i+1)*step], axis=0) for i in range(len(phonemes))]

    _, path = fastdtw(mfcc, target_seq, dist=lambda x, y: np.linalg.norm(x - y))
    alignment = [(mfcc_i, phonemes[phoneme_i]) for mfcc_i, phoneme_i in path]
    return alignment


def print_alignment(alignment, sr, hop_length=512):
    print("Phoneme alignment:")
    printed = set()
    for frame_i, phoneme in alignment:
        if (frame_i, phoneme) not in printed:
            time = frame_i * hop_length / sr
            print(f"{phoneme} @ {time:.2f} sec")
            printed.add((frame_i, phoneme))


def plot_mfcc(mfcc, alignment, sr, hop_length=512):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfcc.T, x_axis='time', sr=sr, hop_length=hop_length)
    times = [frame * hop_length / sr for frame, _ in alignment]
    labels = [phoneme for _, phoneme in alignment]

    for t, l in zip(times, labels):
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        plt.text(t, mfcc.shape[1]-1, l, color='white', rotation=90, verticalalignment='top')

    plt.title("MFCC with phoneme alignment")
    plt.tight_layout()
    plt.show()
