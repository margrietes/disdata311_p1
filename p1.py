import pandas as pd
import tensorflow 
from tensorflow import keras
from keras import layers

# Import data

train = pd.read_csv("/Users/gabriela/Documents/UCR/Spring 2026/Neural Networks & Deep Learning/p1/airbnb/airbnb_train.csv")
X_train = train.drop("price", axis=1)
y_train = train["price"]
# print(len(X_train.columns)) # = 15

valid = pd.read_csv("/Users/gabriela/Documents/UCR/Spring 2026/Neural Networks & Deep Learning/p1/airbnb/airbnb_valid.csv")
X_valid = valid.drop("price", axis=1)
y_valid = valid["price"]

X_test = pd.read_csv("/Users/gabriela/Documents/UCR/Spring 2026/Neural Networks & Deep Learning/p1/airbnb/airbnb_test.csv")

model = keras.Sequential(
    [
        layers.Dense(15, activation="sigmoid"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=512,
    validation_split=0.2,
)