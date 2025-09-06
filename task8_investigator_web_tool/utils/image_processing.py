import cv2
import numpy as np
from PIL import Image

def crop_image(image, coords):
    x1, y1, x2, y2 = coords
    return image[y1:y2, x1:x2]

def select_region(mask, coords):
    x1, y1, x2, y2 = coords
    mask[y1:y2, x1:x2] = 255
    return mask

def deselect_region(mask, coords):
    x1, y1, x2, y2 = coords
    mask[y1:y2, x1:x2] = 0
    return mask

def apply_blackout(image, mask, opacity=1.0):
    overlay = np.zeros_like(image)
    overlay[mask == 255] = 0
    return cv2.addWeighted(image, 1-opacity, overlay, opacity, 0)

def apply_blur(image, mask, intensity=15):
    blurred = cv2.GaussianBlur(image, (intensity*2+1, intensity*2+1), 0)
    result = image.copy()
    result[mask == 255] = blurred[mask == 255]
    return result
