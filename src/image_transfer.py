from PIL import Image, ImageOps

def transfer(image):
    img = Image.open(image)
    gray_img = ImageOps.grayscale(img)
    return gray_img


