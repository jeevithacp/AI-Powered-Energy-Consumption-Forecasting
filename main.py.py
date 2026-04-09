import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\JEEVITHA\Downloads\AI_Energy_Forecasting_Project\data\energy.csv")
df['Date'] = pd.to_datetime(df['Date'])

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

X = df[['Day', 'Month']]
y = df['Consumption']

model = RandomForestRegressor()
model.fit(X, y)

df['Predicted'] = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, df['Predicted']))
r2 = r2_score(y, df['Predicted'])

print("RMSE:", rmse)
print("R2 Score:", r2)

plt.plot(df['Date'], y, label='Actual')
plt.plot(df['Date'], df['Predicted'], label='Predicted')
plt.legend()
plt.savefig("output.png")
plt.show()
