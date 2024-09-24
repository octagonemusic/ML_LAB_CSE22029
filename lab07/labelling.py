import pandas as pd

# Load the data from the Excel sheet, specifying that the column names are in the 6th row (index 5)
file_path = 'groundwater_data.xlsx'
df = pd.read_excel(file_path, header=5)

# Convert relevant columns to numeric, coercing errors to NaN
df['Sodium (mg/L)'] = pd.to_numeric(df['Sodium (mg/L)'], errors='coerce')
df['Bicarbonate (mg/L)'] = pd.to_numeric(df['Bicarbonate (mg/L) '], errors='coerce')
df['Fluoride (mg/L)'] = pd.to_numeric(df['Fluoride (mg/L)'], errors='coerce')
df['Chloride (mg/L)'] = pd.to_numeric(df['Chloride (mg/L)'], errors='coerce')

# Define your thresholds for labeling
chloride_threshold = 250  # Example threshold for chloride
sodium_threshold = 200     # Example threshold for sodium
bicarbonates_threshold = 150  # Example threshold for bicarbonates
fluorides_threshold = 1.0  # Example threshold for fluorides

# Function to determine safety
def label_quality(row):
    if (row['Chloride (mg/L)'] > chloride_threshold or
        row['Sodium (mg/L)'] > sodium_threshold or
        row['Bicarbonate (mg/L)'] < bicarbonates_threshold or
        row['Fluoride (mg/L)'] > fluorides_threshold):
        return "Unsafe"
    else:
        return "Safe"

# Apply the labeling function to each row
df['Quality'] = df.apply(label_quality, axis=1)

# Save the updated DataFrame back to an Excel file
output_file_path = 'labeled_data.xlsx'  # Replace with desired output file path
df.to_excel(output_file_path, index=False)

print("Labeling complete! The data has been saved to", output_file_path)
