# required imports
from flask import Flask, request, jsonify
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from keras.models import load_model

app = Flask(__name__)

model = load_model("my_model_v12.h5")


# load necessary model
def load_ai_model():
    global model
    model = load_model("my_model_v12.h5")


@app.route("/")
def hello():
    return jsonify("Welcome to the PIRM API - Belgian endive edition")


# change img format to required size
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((32, 32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = preprocess_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image).tolist()
            #diseases = ['Aardappelziekte', 'Alternia', 'Bladvlekkenziekte', 'Cladosporium', 'Pepinomosaicvirus',
            #            'Spintmijt', 'Yellow leaf curl']
            diseases = ['green_peach', 'leafminers']

    # return the data dictionary as a JSON resp
    return jsonify(preds, diseases)


# app starting point, loaded model before starting
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
