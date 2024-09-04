from transformers import pipeline
import numpy as np
from PIL import Image
from pathlib import Path


def classify(image_url):
    """
    Classify the age of an image using the ViT model.
    """
    classifier = pipeline(model="nateraw/vit-age-classifier")
    return classifier(image_url)


def preprocess_image(image):
    """
    Preprocess the image for the Vision Transformer (ViT) model.

    Parameters:
    - image (np.ndarray | PIL.Image.Image | str | Path | None): The input image.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    if isinstance(image, str) or isinstance(image, Path):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    return image
