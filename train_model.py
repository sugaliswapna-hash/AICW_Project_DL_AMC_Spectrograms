import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "spectrograms",
    target_size=(64,64),
    batch_size=4,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    "spectrograms",
    target_size=(64,64),
    batch_size=4,
    class_mode="binary",
    subset="validation"
)
model = Sequential([
Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)),
MaxPooling2D(2,2),

Conv2D(64,(3,3),activation="relu"),
MaxPooling2D(2,2),

Flatten(),
Dense(128,activation="relu"),
Dense(1,activation="sigmoid")
])

model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"]
)

model.fit(train,epochs=10,validation_data=val)

if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/cnn_model.h5")

print("Model saved")