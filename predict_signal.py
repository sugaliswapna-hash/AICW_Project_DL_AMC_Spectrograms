import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from tensorflow.keras.models import load_model
import cv2

model = load_model("models/cnn_model.h5")

signal = np.array([0.9,-0.8,0.95,-0.91,0.88,-0.85])

f,t,Sxx = spectrogram(signal)

plt.figure()
plt.pcolormesh(t,f,Sxx)
plt.axis("off")
plt.savefig("test.png",bbox_inches='tight',pad_inches=0)
plt.close()

img = cv2.imread("test.png")
img = cv2.resize(img,(64,64))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = model.predict(img)

if prediction > 0.5:
    print("Predicted Signal : QPSK")
else:
    print("Predicted Signal : BPSK")