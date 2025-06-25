import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv("C:/Users/vishakh/Desktop/ict/datas/beer-servings.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["continent"] = le.fit_transform(df["continent"].astype(str))

X = df.drop(columns=["country", "total_litres_of_pure_alcohol"])
y = df["total_litres_of_pure_alcohol"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))

print(" Model saved as model.pkl")
