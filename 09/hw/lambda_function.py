import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="model_2024_hairstyle_v2.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img

def lambda_handler(event, context):
    url = event["url"]
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    img_array = np.array(img, dtype="float32") / 255.0

    interpreter.set_tensor(input_index, [img_array])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return float(preds[0])
