
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("fake_currency_model.h5")

img = cv2.imread("test_note.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
result = np.argmax(prediction)

if result == 0:
    print("Genuine Currency")
else:
    print("Fake Currency")
