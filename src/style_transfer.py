import numpy as np
from PIL import Image, ImageOps


def transfer_image(image_file_or_path, style):
    """
    Reads an image file and applies style transformation.

    Args:
        image_file_or_path (str): Location of file to style
        style (str): Name of style/model to apply

    Returns:
        Stylized PIL Image
    """
    image = Image.open(image_file_or_path)

    if style == "Grayscale":
        image = ImageOps.grayscale(image)

    return image


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
