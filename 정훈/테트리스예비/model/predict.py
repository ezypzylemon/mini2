import numpy as np
from tensorflow.keras.models import load_model

class ActionClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def classify_action(self, landmarks):
        landmarks = np.array(landmarks).flatten().reshape(1, -1)
        prediction = self.model.predict(landmarks)
        return np.argmax(prediction)

if __name__ == "__main__":
    classifier = ActionClassifier("path/to/model.h5")
    dummy_landmarks = np.random.rand(34)
    action = classifier.classify_action(dummy_landmarks)
    print(f"Predicted action: {action}")
