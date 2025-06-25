import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv("bmi-landmarks.csv")
X = df[['PAR', 'FWHR', 'CJWR']]
y = df['BMI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(df.select_dtypes(include='number').corr())
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

with open("../models/bmi_model.pkl", "wb") as f:
    pickle.dump(model, f)

