# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas

2. Import Decision tree classifier

3. Fit the data in the model

4. Find the accuracy score 

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KOMALAVARSHINI.S
RegisterNumber:  212224230133

```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


data = pd.read_csv("/content/3b87d512-bda4-4173-8465-0df14626dc9f.csv")

# Display top rows
print("🔹 Data Head:")
display(data.head())


print("\n🔹 Data Info:")
data.info()


print("\n🔹 Missing Values:")
print(data.isnull().sum())


le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

print("\n🔹 Encoded Data Head:")
display(data.head())


target_col = [col for col in data.columns if col.lower() == 'salary'][0]
print(f"\n✅ Target Column Detected: {target_col}")


X = data.drop(target_col, axis=1)
y = data[target_col]


regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)


y_pred = regressor.predict(X)


print("\n🔹 Model Evaluation:")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.4f}")


compare_df = pd.DataFrame({'Level': X['Level'], 'Actual Salary': y, 'Predicted Salary': y_pred})
print("\n🔹 Actual vs Predicted Salaries:")
display(compare_df)


plt.figure(figsize=(12,6))
plot_tree(regressor, feature_names=X.columns, filled=True, fontsize=8)
plt.title("Decision Tree Regressor - Employee Salary Prediction")
plt.show()


plt.figure(figsize=(7,5))
plt.scatter(X['Level'], y, color='blue', label='Actual Salary')
plt.plot(X['Level'], y_pred, color='red', linewidth=2, label='Predicted Salary')
plt.xlabel("Employee Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression - Level vs Salary")
plt.legend()
plt.grid(True)
plt.show()
```
## Output:
### Data Head:
![alt text](image.png)

### Data Info:
![alt text](image-1.png)

### Missing Values:
![alt text](image-2.png)

### Encoded Data Head:
![alt text](image-3.png)

### Model Evaluation:
![alt text](image-5.png)

### Actual vs Predicted Salaries:
![alt text](image-6.png)

### Employee Salary Prediction:
![alt text](image-8.png)

### Graph:
![alt text](image-9.png)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
