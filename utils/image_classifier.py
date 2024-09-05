from transformers import pipeline
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import requests
from io import BytesIO


def classify(image_url):
    """
    Classify the age of an image using the ViT model.
    """
    classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
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


def validate_image_url(image_url):
    if image_url is None:
        raise ValueError("Input image cannot be None.")

    try:
        if isinstance(image_url, str) or isinstance(image_url, Path):
            if str(image_url).startswith(("http://", "https://")):
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_url)

    except (FileNotFoundError, UnidentifiedImageError) as err:
        raise ValueError(f"Unable to load the image: {err}")
    except requests.exceptions.RequestException as err:
        raise ValueError(f"Unable to access the URL: {err}")

    return image
