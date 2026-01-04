from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load correct model (keras format)
model = load_model("fake_currency_model.h5")


# ✅ Class mapping (MUST match training)
class_names = ['fake', 'real']

history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["GET", "POST"])
def check():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            pred_index = np.argmax(pred)
            confidence = round(float(pred[0][pred_index]) * 100, 2)

            pred_class = class_names[pred_index]

            if pred_class == "real":
                result = "Genuine Currency"
            else:
                result = "Fake Currency"

            history.append({
                "image": file.filename,
                "result": result,
                "confidence": confidence,
                "time": datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
            })

    return render_template(
    "check.html",
    result=result,
    confidence=confidence,
    image_path=image_path,
    history=history
)


@app.route("/gallery")
def gallery():
    gallery_folder = os.path.join(app.static_folder, "images/gallery")
    images = os.listdir(gallery_folder)
    images = [img for img in images if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    return render_template("gallery.html", images=images)

@app.route("/history")
def history_page():
    return render_template("history.html", history=history)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
