"""
Frequency-domain visualization of an image.

GAN and diffusion upsampling can leave periodic artifacts in the frequency
domain that aren't visible in the raw pixels. This module renders the log-
magnitude spectrum so it can be shown alongside the model's verdict.

Important: this is a fixed signal-processing visualization, not a trained
classifier - it is not used to compute the verdict, only to support it
visually. See README limitations.
"""
import numpy as np
from PIL import Image


def compute_fft_spectrum(image: Image.Image, size: int = 224) -> Image.Image:
    gray = image.convert("L").resize((size, size))
    arr = np.array(gray, dtype=np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.log1p(np.abs(spectrum))

    magnitude -= magnitude.min()
    peak = magnitude.max()
    if peak > 0:
        magnitude /= peak

    return Image.fromarray((magnitude * 255).astype(np.uint8))
