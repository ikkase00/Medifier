import threading
import wave
import tkinter as tk
from tkinter import messagebox, filedialog

import numpy as np
import torch

try:
    import simpleaudio as sa
except Exception:
    sa = None

from LDM import *
from VAE import VAEDecoder
from utils import spec_to_wav

SAMPLE_RATE = 16000

diffusion_core = UNet()
decoder = VAEDecoder(1, 16)

@torch.no_grad()
def generate_audio(
        unet,
        vae_decoder,
        scheduler,
        vocoder=spec_to_wav,
        *,
        num_steps=50,
        batch_size=1,
        latent_shape=(8, 64, 128),
        device="cuda",
        dtype=torch.float16,
        guidance_scale=None
):
    C, F, T = latent_shape
    x = torch.randn(batch_size, C, F, T, device=device, dtype=dtype)
    scheduler.set_timesteps(num_steps, device=device)
    for t in scheduler.timesteps:
        eps = unet(x, t)
        step = scheduler.step(model_output=eps, timestep=t, sample=x)
        x = step.prev_sample
    mel = vae_decoder(x)
    waveform = spec_to_wav(mel, SAMPLE_RATE)
    waveform = waveform.squeeze(1).clamp(-1, 1)
    return waveform


def model_wave_to_pcm_bytes(waveform_tensor):
    y = waveform_tensor[0].detach().float().cpu().numpy()
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16).tobytes()
    return pcm


def save_wave(filename, pcm_bytes):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def prompt_save_dialog(root, pcm_bytes):
    if not pcm_bytes:
        messagebox.showerror("Nothing to save", "No audio data was generated.")
        return
    path = filedialog.asksaveasfilename(
        parent=root,
        title="Save audio file",
        defaultextension=".wav",
        filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        initialfile="output.wav"
    )
    if not path:
        return
    try:
        save_wave(path, pcm_bytes)
        messagebox.showinfo("Saved", f"Audio saved to:\n{path}")
    except Exception as e:
        messagebox.showerror("Save failed", f"Couldn't save file:\n{e}")


def do_work_play_then_offer_save(root, status_label, button):
    try:
        button.config(state=tk.DISABLED)
        status_label.config(text="Generating audio…")
        waveform = generate_audio(
            diffusion_core,
            decoder,
            vocoder = spec_to_wav(),
            num_steps=50,
            batch_size=1,
            latent_shape=(8, 64, 128),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pcm = model_wave_to_pcm_bytes(waveform)
        if sa is None:
            status_label.config(text="Playback skipped. Choose a path to save.")
            root.after(0, prompt_save_dialog, root, pcm)
            return
        status_label.config(text="Playing audio…")
        play_obj = sa.play_buffer(pcm, num_channels=1, bytes_per_sample=2, sample_rate=SAMPLE_RATE)
        play_obj.wait_done()
        status_label.config(text="Choose a path to save.")
        root.after(0, prompt_save_dialog, root, pcm)
    finally:
        button.config(state=tk.NORMAL)
        status_label.config(text="Done. Click to play again.")


def on_click(root, status_label, button):
    threading.Thread(
        target=do_work_play_then_offer_save,
        args=(root, status_label, button),
        daemon=True
    ).start()


def main():
    root = tk.Tk()
    root.title("Play → Save WAV")
    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack()
    status_label = tk.Label(frame, text="Click to play, then choose a path to save.")
    status_label.pack(pady=(0, 10))
    play_button = tk.Button(
        frame,
        text="Play & Save",
        width=22,
        command=lambda: on_click(root, status_label, play_button)
    )
    play_button.pack()
    root.lift()
    root.attributes("-topmost", True)
    root.after(0, root.attributes, "-topmost", False)
    root.mainloop()


if __name__ == "__main__":
    main()
