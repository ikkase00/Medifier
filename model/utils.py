import torchaudio.transforms as T
import matplotlib.pyplot as plt
from librosa.feature import inverse
from librosa.feature import melspectrogram
import librosa
import torch
import sounddevice as sd
import numpy as np

# This file copies function from the "diff.ipynb" notebook to make sure they can be imported as helper functions

# WavToSpec was recreated as a class to work with the pytorch dataset transformation
class WavToSpec:
    def __init__(self, T=512, HOP_LENGTH=512, N_FFT=2048, POWER=2.0):
        self.T = T
        self.HOP_LENGTH = HOP_LENGTH
        self.N_FFT = N_FFT
        self.POWER = POWER

    def __call__(self, wav_path):
        waveform, sample_rate = librosa.load(wav_path)

        target_length = self.T * self.HOP_LENGTH - 1
        if len(waveform) < target_length:
            waveform = np.pad(
                waveform, (0, target_length - len(waveform)), mode="constant"
            )
        else:
            waveform = waveform[:target_length]

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            power=self.POWER,
            n_mels=80,
        )

        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

# Same as in diff.ipynb
def show_spec(mel):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel Bands")
    plt.tight_layout()
    plt.show()

# Same as in diff.ipynb
def spec_to_wav(spec, sample_rate, T=512, HOP_LENGTH=512, N_FFT=2048, POWER=2.0):
    spec = spec
    inv = inverse.mel_to_audio(
        spec,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=POWER,
        n_iter=64
    )

    sd.play(inv, sample_rate)
