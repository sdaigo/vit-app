from flask import Flask, render_template, url_for, request
from matplotlib import image
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
from pytorch_pretrained_vit import ViT
import json

app = Flask(__name__)

app.config['MODEL'] = ViT('B_16_imagenet1k', pretrained=True)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/upload", methods=["post"])
def upload():
    img = request.files["image"]
    filename = secure_filename(img.filename)
    path = "./static/img/{}".format(filename)
    img.save(path)
    return render_template("index.html", saved_image_path=path)


@app.route("/recognition", methods=["post"])
def recognition():
    image_path = request.form['image_path']
    img = Image.open(image_path)

    # 前処理
    tfms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    img = tfms(img)
    img = img.unsqueeze(0)

    print(img.shape)

    with torch.no_grad():
        outputs = app.config["MODEL"](img)

    pred = torch.argmax(outputs)

    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[key] for key in labels_map]
    label = labels_map[pred]
    return render_template("index.html", saved_image_path=image_path, predicted_label=label)


if __name__ == "__main__":
    app.run(debug=True)
