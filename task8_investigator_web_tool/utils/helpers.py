import os
import pandas as pd
from PIL import Image
import numpy as np

def save_image(image, path):
    Image.fromarray(image).save(path)

def log_edit(log_file, image_name, objects_edited, tools_used):
    df = pd.DataFrame([[image_name, ", ".join(objects_edited), ", ".join(tools_used)]],
                      columns=["image_name", "objects_edited", "tools_used"])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

def generate_preview(original, edited):
    original_pil = Image.fromarray(original)
    edited_pil = Image.fromarray(edited)
    width = original_pil.width + edited_pil.width
    height = max(original_pil.height, edited_pil.height)
    preview = Image.new('RGB', (width, height))
    preview.paste(original_pil, (0, 0))
    preview.paste(edited_pil, (original_pil.width, 0))
    return np.array(preview)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
