
from PIL import Image, ImageEnhance, ImageFilter

def apply_dreamy_effects(image_path, mode="lucid"):
    img = Image.open(image_path)

    if mode == "lucid":
        img = img.filter(ImageFilter.GaussianBlur(1.2))
        img = ImageEnhance.Color(img).enhance(1.5)
        img = ImageEnhance.Brightness(img).enhance(1.1)
    elif mode == "lsd":
        img = img.filter(ImageFilter.DETAIL)
        img = ImageEnhance.Color(img).enhance(2.0)
        img = ImageEnhance.Contrast(img).enhance(1.5)
    elif mode == "dreamy":
        img = img.filter(ImageFilter.GaussianBlur(2.5))
        img = ImageEnhance.Brightness(img).enhance(1.3)
        img = ImageEnhance.Color(img).enhance(1.2)
    return img
