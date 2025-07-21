
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

#model = keras.models.load_model("crop_yield_model.keras")
model = load_model('.3986.keras')
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
train_mse = model.history['mae']  
val_mse = model.history['val_mae']  

# Plot MSE over epochs
plt.figure(figsize=(8, 5))
plt.plot(train_mse, label='Training MSE', marker='o')
plt.plot(val_mse, label='Validation MSE', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Model Training Progress')
plt.legend()
plt.grid(True)
plt.show()


y_pred = model.predict(X_test)
ape = np.abs((y_pred.flatten() - y_test) / y_test) * 100  
within_10_percent = np.sum(ape <= 10) 
within_20_percent = np.sum(ape <= 20) 
percentage_within_20 = (within_20_percent / len(y_test)) * 100 
percentage_within_10 = (within_10_percent / len(y_test)) * 100

print(f"Percentage of predictions within 10% of actual: {percentage_within_10:.2f}%")
if percentage_within_10 >= 80:
    print("80% accuracy within 10% threshold")
else:
    print("No 80% accuracy within 10%")
print(f"Percentage of predictions within 20% of actual: {percentage_within_20:.2f}%")
if percentage_within_20 >= 80:
    print("80% accuracy within 20% threshold")
else:
    print("No 80% accuracy within 20%") 