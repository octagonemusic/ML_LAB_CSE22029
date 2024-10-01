import pandas as pd
import scipy.stats as stats

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)


# Function to calculate prior probabilities
def calculate_prior_probabilities(df):
    prior_yes = len(df[df['buys_computer'] == 'yes']) / len(df)
    prior_no = len(df[df['buys_computer'] == 'no']) / len(df)
    return prior_yes, prior_no


# Function to calculate conditional probabilities for a given feature
def conditional_probabilities(df, feature, feature_value, label_value):
    subset = df[df['buys_computer'] == label_value]
    return len(subset[subset[feature] == feature_value]) / len(subset)


# Function to print all conditional probabilities
def print_conditional_probabilities(df):
    for feature in df.columns[:-1]:  # Exclude 'buys_computer'
        print(f"\nConditional probabilities for feature '{feature}':")
        for value in df[feature].unique():
            p_yes = conditional_probabilities(df, feature, value, 'yes')
            p_no = conditional_probabilities(df, feature, value, 'no')
            print(f"P({feature} = {value} | buys_computer = yes): {p_yes:.3f}")
            print(f"P({feature} = {value} | buys_computer = no): {p_no:.3f}")


# Function to perform Chi-square test for independence between two features
def chi_square_test(df, feature1, feature2):
    contingency_table = pd.crosstab(df[feature1], df[feature2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p, dof, expected


# Function to print Chi-square test results for multiple pairs of features
def test_independence(df, feature_pairs):
    for feature1, feature2 in feature_pairs:
        print(f"\nChi-square test between '{feature1}' and '{feature2}':")
        chi2, p, dof, expected = chi_square_test(df, feature1, feature2)
        print(f"Chi-square statistic: {chi2:.3f}")
        print(f"P-value: {p:.3f}")
        print(f"Degrees of freedom: {dof}")
        print("Expected frequencies:")
        print(expected)


# Main function to run all the calculations
def main():
    # 1. Calculate and print prior probabilities
    prior_yes, prior_no = calculate_prior_probabilities(df)
    print(f"Prior Probability P(buys_computer = yes): {prior_yes:.3f}")
    print(f"Prior Probability P(buys_computer = no): {prior_no:.3f}")

    # 2. Print conditional probabilities
    print_conditional_probabilities(df)

    # 3. Perform Chi-square test for independence
    feature_pairs = [('age', 'income'), ('student', 'credit_rating'), ('age', 'student')]
    test_independence(df, feature_pairs)


# Execute main function
if __name__ == "__main__":
    main()
