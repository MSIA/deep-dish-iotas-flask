import numpy as np
from PIL import Image, ImageOps

from src import evaluate


def transfer_image(image_file_or_path, style, save_path):
    """
    Reads an image file and applies style transformation.

    Args:
        image_file_or_path (str): Location of file to style
        style (str): Name of style/model to apply

    Returns:
        Stylized PIL Image
    """
    image = Image.open(image_file_or_path)

    # Simple PIL Image transformations (no model used)
    if style == "Grayscale":
        image = ImageOps.grayscale(image)
        image.save(save_path)

    # Use pretrained style transfer models
    DEVICE = "/CPU:0"
    BATCH_SIZE = 1
    image = np.clip(np.array(image), 0, 255).astype(np.uint8)

    if style == "Wave":
        image = evaluate.ffwd(
            image,
            save_path,
            "src/models/wave.ckpt",
            device_t=DEVICE,
            batch_size=BATCH_SIZE
        )


def transfer_video_frame(frame, style):
    """
    Stylizes a video frame.

    Args:
        frame (PIL.Image): Image from `camera.read()`
        style (str): Name of style/model to apply

    Returns:
        numpy.array of stylized image
    """
    # Convert to PIL Image to use their transformations
    frame = Image.fromarray(frame)

    # Stylize
    if style == "Grayscale":
        frame = ImageOps.grayscale(frame)

    # Convert back to NumPy array for buffer stream
    # Note that this is needed for video feeds but not static images
    frame = np.array(frame)

    return frame
