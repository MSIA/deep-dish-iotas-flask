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
        None (saves the resulting image to file)
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

    if style == "La Muse":
        model = "src/models/la_muse.ckpt"
    elif style == "Rain Princess":
        model = "src/models/rain_princess.ckpt"
    # elif style == "Starry Night":
    #     model = "src/models/starry_night/"
    elif style == "The Scream":
        model = "src/models/scream.ckpt"
    elif style == "Udnie":
        model = "src/models/udnie.ckpt"
    elif style == "Wave":
        model = "src/models/wave.ckpt"
    elif style == "Wreck":
        model = "src/models/wreck.ckpt"

    evaluate.ffwd(
        image_in=image,
        save_path=save_path,
        saved_model=model,
        device_t=DEVICE,
        batch_size=BATCH_SIZE,
        save=True
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

    # Simple PIL Image transformations (no model used)
    if style == "Grayscale":
        frame = ImageOps.grayscale(frame)
        frame = np.array(frame)
        return frame

    # Use pretrained style transfer models
    DEVICE = "/CPU:0"
    BATCH_SIZE = 1
    frame = np.clip(np.array(frame), 0, 255).astype(np.uint8)

    if style == "Wave":
        model = "src/models/wave.ckpt"

    frame = evaluate.ffwd_video(
        image_in=frame,
        save_path="",
        saved_model=model,
        device_t=DEVICE,
        batch_size=BATCH_SIZE,
        save=False
    )

    # Convert back to NumPy array for buffer stream
    frame = np.array(frame)

    return frame
