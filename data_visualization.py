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


# Save the updated dataset to a new CSV file
data.to_csv('updated_datasetforassignment2.csv', index=False)

# Calculate the average Composite_Index for Gender
gender_avg = data.groupby('Gender')['Composite_Index'].mean()
# Calculate the average Composite_Index for Location
location_avg = data.groupby('Location')['Composite_Index'].mean()  
# Visualize Composite Index vs Age
bins = range(10, 70, 10)
labels = [f"{i}-{i+9}" for i in bins[:-1]]
data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# Calculate the average Composite_Index for each age group
age_group_avg = data.groupby('Age Group')['Composite_Index'].mean()

# Plot the data
plt.figure(figsize=(10, 6))
age_group_avg.plot(kind='bar', color='skyblue', edgecolor='black')
ax = age_group_avg.plot(kind='bar', color='skyblue', edgecolor='black')

# Add values on top of each bar
for i, v in enumerate(age_group_avg):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')



# Customize the chart
plt.title('Average Composite Index by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Composite Index')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Visualize Composite Index vs Gender

plt.figure(figsize=(10, 5))
ax1 = gender_avg.plot(kind='bar', color='lightcoral', edgecolor='black')
ax1.set_title('Average Composite Index by Gender')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Average Composite Index')
ax1.set_xticklabels(['Male', 'Female'], rotation=0)

# Add values on top of each bar for Gender
for i, v in enumerate(gender_avg):
    ax1.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Visualize Composite Index vs Location

plt.figure(figsize=(10, 5))
ax2 = location_avg.plot(kind='bar', color='lightseagreen', edgecolor='black')
ax2.set_title('Average Composite Index by Location')
ax2.set_xlabel('Location')
ax2.set_ylabel('Average Composite Index')
# ax2.set_xticklabels(['Location 1', 'Location 2', 'Location 3'], rotation=0)
# Add values on top of each bar for Location
for i, v in enumerate(location_avg):
    ax2.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()
