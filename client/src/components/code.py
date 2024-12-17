import requests
from PIL import Image
from io import BytesIO

api_key = "a6a1e5aa8e8f0c78bc30"
api_secret_key = "063178799291102f8ddb5949fb43015752fd7af75d973ee3e837f18333efc5c1"

headers = {
    'pinata_api_key': api_key,
    'pinata_secret_api_key': api_secret_key
}
url = 'https://api.pinata.cloud/data/pinList?pageLimit=1'
response = requests.get(url, headers=headers)
pins = response.json()
pinata_cid = pins['rows'][0]['ipfs_pin_hash']
# print("Hash id of image: ", pinata_cid)
# replace with your own values
pinata_endpoint = 'https://gateway.pinata.cloud/ipfs/'

# make request to Pinata
response = requests.get(pinata_endpoint + pinata_cid)

# convert response content to PIL Image object
img2 = Image.open(BytesIO(response.content))
# img2 = user_image.convert('RGB').resize((224, 224))

# display image
img2.show()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def lpredictions(model_path: str, user_image: Image.Image) -> None:
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    # Print label name
    label = {0: "benign", 1: "malignant"}
    # Convert the image to RGB format and resize to the correct size
    img_ = user_image.convert('RGB').resize((224, 224))
    # Convert the image to the correct format
    imag = image.img_to_array(img_)
    imag = np.expand_dims(imag, axis=0)
    pred = (model.predict(imag) > 0.5).astype(int)
    # print(pred)
    print(label[pred[0][0]])
    plt.imshow(img_)

# Usage
model_path = "C:\Users\sonia\Downloads\Main_project_3rd_year-master\client\src\components\resnet50v2.keras"
# Load the image file into a PIL Image object
user_image = Image.open(BytesIO(response.content))

lpredictions(model_path, user_image)
