import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from an Excel file."""
    return pd.read_excel(file_path)

def prepare_data(df):
    """Prepare features and target variable from the DataFrame."""
    X = df[['Sodium (mg/L)', 'Bicarbonate (mg/L)', 'Fluoride (mg/L)', 'Chloride (mg/L)']]
    y = df['Quality']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_classifier(X_train, y_train, max_depth=None, criterion='gini'):
    """Train a Decision Tree classifier with optional max_depth and criterion."""
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    """Evaluate the model and print accuracy, classification report, and confusion matrix."""
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'\n{"="*30}')
    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'{"="*30}\n')

    print("Classification Report (Test Data):")
    print(classification_report(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return y_train_pred, y_test_pred

def feature_importance(clf, feature_names):
    """Print the feature importances of the classifier."""
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nFeature Importances:")
    print(feature_importances.sort_values(by='Importance', ascending=False))

def plot_decision_tree(clf, feature_names, title):
    """Plot the decision tree."""
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=feature_names, class_names=['Safe', 'Unsafe'], filled=True)
    plt.title(title)
    plt.show()

def main(file_path):
    """Main function to run the workflow."""
    df = load_data(file_path)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train with max_depth constraint and evaluate
    print("Training Decision Tree with max_depth=5 and default criterion (gini):")
    clf_gini = train_classifier(X_train, y_train, max_depth=5)
    evaluate_model(clf_gini, X_train, y_train, X_test, y_test)
    feature_importance(clf_gini, X.columns)
    plot_decision_tree(clf_gini, X.columns, 'Decision Tree Visualization (max_depth=5, criterion=gini)')

    # Train with max_depth constraint and entropy criterion and evaluate
    print("\nTraining Decision Tree with max_depth=5 and criterion='entropy':")
    clf_entropy = train_classifier(X_train, y_train, max_depth=5, criterion='entropy')
    evaluate_model(clf_entropy, X_train, y_train, X_test, y_test)
    feature_importance(clf_entropy, X.columns)
    plot_decision_tree(clf_entropy, X.columns, 'Decision Tree Visualization (max_depth=5, criterion=entropy)')

if __name__ == "__main__":
    file_path = 'labeled_data.xlsx'  # Replace with your labeled file path
    main(file_path)
