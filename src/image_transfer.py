import numpy as np
from PIL import Image, ImageOps


def transfer_image(image_file_or_path):
    """
    Reads an image file and applies style transformation.

    Args:
        image_file_or_path: Location of file to style

    Returns:
        Stylized PIL Image
    """
    image = Image.open(image_file_or_path)
    gray_img = ImageOps.grayscale(image)
    return gray_img


def transfer_video_frame(frame):
    """
    Stylizes a video frame.

    Args:
        frame: Image from

    Returns:
        numpy.array of stylized image
    """
    # Convert to PIL Image to use their transformations
    frame = Image.fromarray(frame)

    # Stylize
    frame = ImageOps.grayscale(frame)

    # Convert back to NumPy array for buffer stream
    # Note that this is needed for video feeds but not static images
    frame = np.array(frame)

    return frame
