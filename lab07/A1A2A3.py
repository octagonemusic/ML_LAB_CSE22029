import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Creating the dataset
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Calculate entropy of the target variable buys_computer
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = -np.sum([(counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy_value

# Step 3: Function to calculate Information Gain
def information_gain(data, split_attribute_name, target_name="buys_computer"):
    # Calculate the total entropy of the target attribute
    total_entropy = entropy(data[target_name])
    
    # Get unique values and counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    # Weighted entropy
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    # Calculate the Information Gain
    information_gain_value = total_entropy - weighted_entropy
    return information_gain_value

# Step 4: Calculate Information Gain for each attribute
attributes = ['age', 'income', 'student', 'credit_rating']
info_gains = {attribute: information_gain(df, attribute) for attribute in attributes}

# Determine the attribute with the highest Information Gain
best_feature = max(info_gains, key=info_gains.get)

# Print Information Gain and best feature
print("Information Gain for each attribute:")
for attribute, gain in info_gains.items():
    print(f"{attribute}: {gain}")

print(f"\nBest feature to split on: {best_feature}")

# Step 5: Encoding the categorical features and labels
le_age = LabelEncoder()
le_income = LabelEncoder()
le_student = LabelEncoder()
le_credit_rating = LabelEncoder()
le_buys_computer = LabelEncoder()

df['age'] = le_age.fit_transform(df['age'])
df['income'] = le_income.fit_transform(df['income'])
df['student'] = le_student.fit_transform(df['student'])
df['credit_rating'] = le_credit_rating.fit_transform(df['credit_rating'])
df['buys_computer'] = le_buys_computer.fit_transform(df['buys_computer'])

# Features and target
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Step 6: Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 7: Model's training set accuracy
training_accuracy = model.score(X, y)
print(f"Training Accuracy: {training_accuracy}")

# Step 8: Get the depth of the tree
tree_depth = model.get_depth()
print(f"Depth of the tree: {tree_depth}")

# Step 4: Visualize the decision tree with feature names
plt.figure(figsize=(25,15))  # Adjust the size for better spacing
plot_tree(
    model, 
    filled=True, 
    feature_names=['age', 'income', 'student', 'credit_rating'],  # Feature names explicitly passed
    class_names=['No', 'Yes'],  # Class names
    fontsize=12,  # Adjust text size
    rounded=True,  # Rounded corners for a cleaner look
    proportion=True  # Box size proportional to sample size in each node
)
plt.show()                      
        
