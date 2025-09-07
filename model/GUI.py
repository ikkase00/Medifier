import threading
import wave
import tkinter as tk
from tkinter import messagebox, filedialog

import numpy as np
import torch
sa = None
from LDM import *
from VAE import VAEDecoder
from utils import spec_to_wav

# Initialize parameters and models
SAMPLE_RATE = 16000

diffusion_core = UNet() # Initialize diffusion core for generating latent space
decoder = VAEDecoder(1, 16) # Initialize the decoder to decode latent space into mel spectrogram


# Genenrates mel spectrogram from my model
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
    C, F, T = latent_shape # Initialize the target latent tensor for the scheduler
    x = torch.randn(batch_size, C, F, T, device=device, dtype=dtype) # Generate random noise as the initial latent representation

    # Prepare scheduler
    scheduler.set_timesteps(num_steps, device=device)

    # Iteratively denoise the latent space
    for t in scheduler.timesteps:
        eps = unet(x, t) # Use the diffusion core to predict noise
        step = scheduler.step(model_output=eps, timestep=t, sample=x) # Update step
        x = step.prev_sample # Get the sample after deniosing

    # After the iterative process, decode the final latent representation into mel spectrogram
    mel = vae_decoder(x)

    # Convert mel into audio wave
    waveform = spec_to_wav(mel, SAMPLE_RATE)
    waveform = waveform.squeeze(1).clamp(-1, 1)
    return waveform


# The GUI below was wrote with a lot of help from chatgpt as I have very little knowledge about GUI and time
# doesn't allow me to learn from scratch. Comments of the GUI are mostly written by chatgpt too.
def model_wave_to_pcm_bytes(waveform_tensor):
    # Convert waveform tensor into PCM-encoded bytes (16-bit signed integers)
    y = waveform_tensor[0].detach().float().cpu().numpy()
    y = np.clip(y, -1.0, 1.0)  # Clamp values to [-1, 1]
    pcm = (y * 32767.0).astype(np.int16).tobytes()
    return pcm


def save_wave(filename, pcm_bytes):
    # Save PCM bytes as a WAV audio file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)        # Mono audio
        wf.setsampwidth(2)        # 2 bytes per sample (16-bit)
        wf.setframerate(SAMPLE_RATE)  # Sampling rate
        wf.writeframes(pcm_bytes)     # Write audio frames


def prompt_save_dialog(root, pcm_bytes):
    # Open save dialog to let user choose where to save audio
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

    if not path:  # User cancelled save
        return

    try:
        save_wave(path, pcm_bytes)
        messagebox.showinfo("Saved", f"Audio saved to:\n{path}")
    except Exception as e:
        messagebox.showerror("Save failed", f"Couldn't save file:\n{e}")


def do_work_play_then_offer_save(root, status_label, button):
    # Run audio generation → playback → prompt to save
    try:
        button.config(state=tk.DISABLED)
        status_label.config(text="Generating audio…")

        # Generate audio with diffusion model
        waveform = generate_audio(
            diffusion_core,
            decoder,
            vocoder=spec_to_wav(),  # Convert spectrogram → waveform
            num_steps=50,
            batch_size=1,
            latent_shape=(8, 64, 128),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Convert tensor → PCM bytes
        pcm = model_wave_to_pcm_bytes(waveform)

        if sa is None:  # If no playback library available
            status_label.config(text="Playback skipped. Choose a path to save.")
            root.after(0, prompt_save_dialog, root, pcm)
            return

        # Play audio
        status_label.config(text="Playing audio…")
        play_obj = sa.play_buffer(pcm, num_channels=1, bytes_per_sample=2, sample_rate=SAMPLE_RATE)
        play_obj.wait_done()

        # After playback, offer to save
        status_label.config(text="Choose a path to save.")
        root.after(0, prompt_save_dialog, root, pcm)

    finally:
        # Re-enable button after work completes
        button.config(state=tk.NORMAL)
        status_label.config(text="Done. Click to play again.")


def on_click(root, status_label, button):
    # Run audio generation in a background thread (non-blocking GUI)
    threading.Thread(
        target=do_work_play_then_offer_save,
        args=(root, status_label, button),
        daemon=True
    ).start()


def main():
    # Build GUI window
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

    # Bring window to front
    root.lift()
    root.attributes("-topmost", True)
    root.after(0, root.attributes, "-topmost", False)

    root.mainloop()


if __name__ == "__main__":
    main()
