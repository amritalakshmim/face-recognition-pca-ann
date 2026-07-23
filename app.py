import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.predict import FacePredictor

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

predictor = FacePredictor()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return render_template(
        "index.html",
        error="Please select an image."
    )

    if not allowed_file(file.filename):
        return render_template(
        "index.html",
        error="Only JPG, JPEG and PNG images are allowed."
    )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    prediction, confidence = predictor.predict(filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)