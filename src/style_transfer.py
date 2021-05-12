import numpy as np
from PIL import Image, ImageOps

from src import evaluate


def transfer_image(image_file_or_path, style, save_path):
    """
    Reads an image file and applies style transformation.

    Args:
        image_file_or_path (str): Location of file to style
        style (str): Name of style/model to apply
        save_path (str): Where to save result image

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

    model = get_model_path_from_name(style)

    evaluate.style_image(
        image_in=image,
        save_path=save_path,
        saved_model=model,
        device_t=DEVICE,
        batch_size=BATCH_SIZE,
        save=True
    )


def transfer_video(video_file_or_path, style, save_path):
    """
    Reads a video file and applies style transformation.

    Args:
        video_file_or_path (str): Location of file to style
        style (str): Name of style/model to apply
        save_path (str): Where to save result video

    Returns:
        None (saves the resulting image to file)
    """
    DEVICE = "/CPU:0"
    BATCH_SIZE = 1

    model = get_model_path_from_name(style)

    evaluate.style_video(
        video_in=video_file_or_path,
        save_path=save_path,
        saved_model=model,
        device_t=DEVICE,
        batch_size=BATCH_SIZE
    )


def transfer_webcam(style):
    """
    Stylizes a video frame.

    Args:
        frame (PIL.Image): Image from `camera.read()`
        style (str): Name of style/model to apply

    Returns:
        numpy.array of stylized image
    """
    DEVICE = "/CPU:0"
    BATCH_SIZE = 1
    model = get_model_path_from_name(style)

    for frame in evaluate.style_webcam(
        saved_model=model,
        device_t=DEVICE,
        batch_size=BATCH_SIZE
    ):
        yield frame


def get_model_path_from_name(model_name):
    if model_name == "La Muse":
        return "src/models/la_muse.ckpt"
    elif model_name == "Rain Princess":
        return "src/models/rain_princess.ckpt"
    # elif model_name == "Starry Night":
    #     return "src/models/starry_night/"
    elif model_name == "The Scream":
        return "src/models/scream.ckpt"
    elif model_name == "Udnie":
        return "src/models/udnie.ckpt"
    elif model_name == "Wave":
        return "src/models/wave.ckpt"
    elif model_name == "Wreck":
        return "src/models/wreck.ckpt"
    else:
        raise Exception
