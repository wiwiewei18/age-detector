from PIL import Image
import requests
from io import BytesIO


def read(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))
