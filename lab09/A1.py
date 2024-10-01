import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from an Excel file."""
    data = pd.read_excel(file_path)
    return data

def clean_data(data, columns):
    """Remove rows with null values in specified columns."""
    cleaned_data = data.dropna(subset=columns)
    return cleaned_data

def create_scatter_plot(data, x_column, y_column):
    """Create and display a scatter plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.5)
    plt.title(f'Scatter Plot of {y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

def main(file_path):
    """Main function to execute the data loading, cleaning, and plotting."""
    # Load the dataset
    data = load_data(file_path)
    
    # Clean the data
    cleaned_data = clean_data(data, ['Sodium (mg/L)', 'Chloride (mg/L)'])
    
    # Create the scatter plot
    create_scatter_plot(cleaned_data, 'Sodium (mg/L)', 'Chloride (mg/L)')

# Execute the script
if __name__ == "__main__":
    main('labeled_data.xlsx')  # Update this path if necessary
