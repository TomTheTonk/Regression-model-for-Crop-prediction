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
# Load Dataset
df = pd.read_csv("crop_yield.csv") 
df.info()
X = df.drop(columns=["Yield_tons_per_hectare"])
y = df["Yield_tons_per_hectare"]


categorical_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
numerical_cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
boolean_cols = ["Fertilizer_Used", "Irrigation_Used"]
# Convert boolean columns to integers (True = 1, False = 0)
X[boolean_cols] = X[boolean_cols].astype(int)


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())  
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])


boolean_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('bool', boolean_transformer, boolean_cols)
    ])


X_processed = preprocessor.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.1, random_state=42)


device = "/GPU:0" if tf.config.experimental.list_physical_devices('GPU') else "/CPU:0"
print(device)
learning_rate = 0.001
with tf.device(device):  
    
    model = keras.Sequential([
        Input(shape=(X_train.shape[1],)),  
        Dense(256), 
        BatchNormalization(),  # Batch Normalization
        ReLU(),  # ReLU activation
        Dense(128),  
        BatchNormalization(),  # Batch Normalization
        ReLU(),  # ReLU activation
        Dense(64),  
        BatchNormalization(),  # Batch Normalization
        ReLU(),  # ReLU activation
        Dense(32),  
        BatchNormalization(),  # Batch Normalization
        ReLU(),  # ReLU activation
        Dense(1)  # Output layer 
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Mean Absolute Error: {mae:.4f} tons per hectare")
model.save("crop_yield_model.keras")
#model = keras.models.load_model("crop_yield_model.keras")

train_mse = history.history['mae']  
val_mse = history.history['val_mae']  

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
# df.hist(bins=50, figsize =(20,10))
# plt.show()
#sns.scatterplot(x='Rainfall_mm',
                #y='Yield_tons_per_hectare',
                #hue = 'Crop',
                #data = df)
#plt.show()
# crops = ['Maize', 'Cotton', 'Rice', 'Wheat', 'Barley', 'Soybean']
# for crop in crops:
#     # Filter the DataFrame to only include the current crop
#     crop_df = df[df['Crop'] == crop]

#     # Create the scatter plot for the current crop
#     sns.scatterplot(x='Rainfall_mm', y='Yield_tons_per_hectare', data=crop_df)

#     # Set the title with the crop name
#     plt.title(f'Scatter Plot for {crop}')

#     # Show the plot
#     plt.show()
# Calculate the average yield for each crop
# Calculate the average yield for each crop
# Filter the DataFrame to only include Rice and Wheat
#rice_wheat_df = df[df['Crop'].isin(['Rice', 'Wheat'])]

# # Calculate the average yield for Rice and Wheat
# average_yield_rice_wheat = rice_wheat_df.groupby('Crop')['Yield_tons_per_hectare'].mean()

# # Print the average yields
# print(average_yield_rice_wheat)
# average_yield = df.groupby('Crop')['Yield_tons_per_hectare'].mean().reset_index()

# # Create a bar plot for the average yield of each crop
# ax = sns.barplot(x='Crop', y='Yield_tons_per_hectare', data=average_yield)

# # Add the values on top of the bars
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.2f}',  # Format the value to 2 decimal places
#                 (p.get_x() + p.get_width() / 2., p.get_height()),  # Position on top of the bar
#                 ha='center', va='center',  # Horizontal and vertical alignment
#                 fontsize=12, color='black',  # Text properties
#                 xytext=(0, 5),  # Text offset (so it's above the bar)
#                 textcoords='offset points')

# # Set the title of the plot
# plt.title('Average Yield per Crop')

# # Show the plot
# plt.show()

