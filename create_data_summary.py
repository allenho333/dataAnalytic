import pandas as pd

# Load the CSV dataset
file_path = './datasetforassignment2.csv'
data = pd.read_csv(file_path)
data['Activity Level'] = data['Activity Level'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2})
# Display the first few rows and summary statistics to inspect the dataset
data.head(), data.describe(include='all')

# Calculate mean, standard deviation, median, and range for numerical columns
numerical_summary = {
    "Mean": data.drop(columns=['User ID']).mean(numeric_only=True),
    "Standard Deviation": data.drop(columns=['User ID']).std(numeric_only=True),
    "Median": data.drop(columns=['User ID']).median(numeric_only=True),
    "Range": data.drop(columns=['User ID']).max(numeric_only=True) - data.min(numeric_only=True)
}

# convert it to excel and save it as summary_statistics.xlsx
numerical_summary_df = pd.DataFrame(numerical_summary)
numerical_summary_df.to_excel('summary_statistics2.xlsx')

