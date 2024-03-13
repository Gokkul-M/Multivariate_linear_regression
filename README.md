# Multivariate_linear_regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Jupter Notebook /Google Colab
## Algorithm:
### Step1
import pandas as pd.
### Step2
Read the csv file.
### Step3
Get the value of X and y variables.
### Step4
Create the linear regression model and fit.
### Step5
Predict the CO2 emission of a car 
## Program:
```
# Program Developed By :Gokkul M
# Register Number: 212223240039
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('FuelConsumption.csv')
plt.figure(figsize=(8, 6))
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='green', label='CYLINDERS vs CO2EMISSIONS')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.title('Scatter Plot: CYLINDERS vs CO2EMISSIONS')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='blue', label='CYLINDERS vs CO2EMISSIONS')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.title('CYLINDERS vs CO2EMISSIONS')
plt.subplot(1, 2, 2)
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='red', label='ENGINESIZE vs CO2EMISSIONS')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.title('ENGINESIZE vs CO2EMISSIONS')
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='blue', label='CYLINDERS vs CO2EMISSIONS')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.title('CYLINDERS vs CO2EMISSIONS')
plt.subplot(1, 3, 2)
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='red', label='ENGINESIZE vs CO2EMISSIONS')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.title('ENGINESIZE vs CO2EMISSIONS')
plt.subplot(1, 3, 3)
plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], color='purple', label='FUELCONSUMPTION vs CO2EMISSIONS')
plt.xlabel('FUELCONSUMPTION (combined)')
plt.ylabel('CO2EMISSIONS')
plt.title('FUELCONSUMPTION vs CO2EMISSIONS')
plt.tight_layout()
plt.show()
X_CYLINDERS = data[['CYLINDERS']]
y_co2 = data['CO2EMISSIONS']
X_train, X_test, y_train, y_test = train_test_split(X_CYLINDERS, y_co2, test_size=0.3, random_state=42)
model_CYLINDERS = LinearRegression()
model_CYLINDERS.fit(X_train, y_train)
CYLINDERS_pred = model_CYLINDERS.predict(X_test)
print(f"Model with CYLINDERS: R2 Score = {r2_score(y_test, CYLINDERS_pred):.4f}")
X_fuel = data[['FUELCONSUMPTION_COMB']]
X_train, X_test, y_train, y_test = train_test_split(X_fuel, y_co2, test_size=0.3, random_state=42)
model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)
fuel_pred = model_fuel.predict(X_test)
print(f"Model with FUELCONSUMPTION: R2 Score = {r2_score(y_test, fuel_pred):.4f}")
ratios = [0.6, 0.7, 0.8]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_CYLINDERS, y_co2, test_size=1 - ratio, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Train-Test Ratio {ratio:.1f}: R2 Score = {r2_score(y_test, pred):.4f}")

```
## Output:
![image](https://github.com/Gokkul-M/Multivariate_linear_regression/assets/144870543/99a83e37-af60-44e7-b935-f47f5a49cf24)
![image](https://github.com/Gokkul-M/Multivariate_linear_regression/assets/144870543/fce32d69-2867-4f46-aa2f-ce8945432f70)
![image](https://github.com/Gokkul-M/Multivariate_linear_regression/assets/144870543/674cd452-43c4-41f9-a2fb-cdf928b55aaf)

```
Model with CYLINDERS: R2 Score = 0.7413
Model with FUELCONSUMPTION: R2 Score = 0.8001
Train-Test Ratio 0.6: R2 Score = 0.7239
Train-Test Ratio 0.7: R2 Score = 0.7413
Train-Test Ratio 0.8: R2 Score = 0.7317
```
## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
