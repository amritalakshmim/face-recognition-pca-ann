from src.predict import FacePredictor

predictor = FacePredictor()

result = predictor.predict(
    "dataset/faces/Aamir/face_5.jpg"
)

print(result)