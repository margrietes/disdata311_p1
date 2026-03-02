import pandas as pd

train = pd.read_csv("/Users/gabriela/Documents/UCR/Spring 2026/Neural Networks & Deep Learning/p1/airbnb/airbnb_train.csv")
X_train = train.drop("price")
y_train = train["price"]

valid = pd.read_csv("/Users/gabriela/Documents/UCR/Spring 2026/Neural Networks & Deep Learning/p1/airbnb/airbnb_valid.csv")
X_valid = valid.drop("price")
y_valid = valid["price"]

X_test = pd.read_csv("/Users/gabriela/Documents/UCR/Spring 2026/Neural Networks & Deep Learning/p1/airbnb/airbnb_test.csv")

