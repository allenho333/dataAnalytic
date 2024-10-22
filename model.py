# import datasetforassignment2.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load the CSV dataset
file_path = './datasetforassignment2.csv'
data = pd.read_csv(file_path)
data['Activity Level'] = data['Activity Level'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2})

# data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
# data['Activity Level'] = data['Activity Level'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2})
# data['Location'] = data['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})


# Z-Score Normalization function
def z_score_normalize(column):
    return (column - column.mean()) / column.std()

# Apply Z-Score Normalization to the performance metrics
data['Normalized_App_Sessions'] = z_score_normalize(data['App Sessions'])
data['Normalized_Distance_Travelled'] = z_score_normalize(data['Distance Travelled (km)'])
data['Normalized_Calories_Burned'] = z_score_normalize(data['Calories Burned'])
data['Normalized_activity_level'] = z_score_normalize(data['Activity Level'])

# Calculate the Composite Index (equal weights)
data['Composite_Index'] = (data['Normalized_App_Sessions'] +
                           data['Normalized_Distance_Travelled'] +
                           data['Normalized_Calories_Burned']
                                                   + data['Normalized_activity_level'])/ 4

# One-Hot Encode categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Location'], drop_first=True)

# create a model to predict the composite index based on gender, activity level, and location



# # Define feature columns and target column
X = data.drop(columns=['User ID','Composite_Index','App Sessions','Distance Travelled (km)','Calories Burned','Activity Level','Normalized_App_Sessions','Normalized_Distance_Travelled','Normalized_Calories_Burned','Normalized_activity_level'])  # Features: Gender, Activity Level, Location
y = data['Composite_Index']  # Target: Composite Index

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_test:",X_test)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

#Get the coefficients (β₁, β₂, ..., βn)
# print('model.coef_:',model.coef_)

# Print the feature names in the same order as the coefficients
feature_names = X_train.columns
coefficients = model.coef_

# Display each feature along with its coefficient
for feature, coef in zip(feature_names, coefficients):
    print(f'{feature}: {coef}')

# Get the intercept (β₀)
# print('intercept (β₀):',model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')







