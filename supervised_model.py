# import matplotlib
# matplotlib.use('Agg')  # Use the non-GUI backend

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from data_simulation import generate_fake_data
from feature_engineering import generate_features

def extract_features(df):
    # Extracting features like sequence length, etc.
    required_columns = ['social_count', 'info_count', 'shopping_count', 'entertainment_count']
    # Example: unique_domains and repeats (based on your data)
    df['unique_domains'] = df.apply(lambda row: len(set([f"domain_{i}" for i in range(int(row['social_count']*10))])), axis=1)
    df['repeats'] = df.apply(lambda row: np.random.randint(1, 5), axis=1)

    # Other features (sequence_length, entropy, etc.)
    df['sequence_length'] = df['social_count'] + df['info_count'] + df['shopping_count'] + df['entertainment_count']
    df['sequence_entropy'] = np.random.uniform(0.0, 1.0, size=df.shape[0])
    df['position_weighted_sum'] = np.random.uniform(0.0, 1.0, size=df.shape[0])
    return df


def generate_fake_data(num_users=100):
    """Generate fake user data with browsing sequences."""
    # Simulating user data
    data = {
        'social_count': np.random.uniform(0, 1, num_users),
        'info_count': np.random.uniform(0, 1, num_users),
        'shopping_count': np.random.uniform(0, 1, num_users),
        'entertainment_count': np.random.uniform(0, 1, num_users),
        'user_id': np.arange(1, num_users + 1)
    }
    
    # Generate age_group (categorical data)
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    data['age_group'] = np.random.choice(age_groups, size=num_users)

    df = pd.DataFrame(data)
    return df


def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve for multiclass classification using One-vs-Rest."""
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)  # Binarize the true labels

    # Check if y_scores is 2D, which means it's the predicted probabilities
    if len(y_scores.shape) == 1:
        raise ValueError("Expected 2D array for predicted probabilities, got 1D")

    # Create OneVsRestClassifier for multiclass ROC
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Compute ROC curve and ROC area for each class
    for i in range(y_true_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    for i in range(y_true_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')  # Save the plot as a file
    plt.close()

def plot_learning_curve(model, X, y):
    """Plot the learning curve to evaluate model performance with more data."""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1,
                                                            train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score", color='b')
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Test score", color='r')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc="best")
    plt.savefig('learning_curve.png')  # Save the plot as a file
    plt.close()

def main():
    print("\n🔍 [MODE] Supervised Prediction Started")

    print("\n📦 Step 1: Generating user data...")
    df = generate_fake_data(num_users=100)
    print("✅ User data generated.\n")

    print("🧪 Step 2: Extracting features...")
    df = extract_features(df)
    print("✅ Features extracted:\n", df.columns)

    # Now df contains 'age_group' from the data generation
    X = df[['sequence_length', 'unique_domains', 'repeats', 'social_count', 'info_count', 'shopping_count', 'entertainment_count']]
    y = df['age_group']  # Now 'age_group' exists in the dataset

    # Label encode the target variable if it's categorical
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    print("📊 Step 3: Performing Stratified 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation with additional metrics
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"✅ Cross-validation completed. Accuracy: {scores.mean():.4f} ± {scores.std():.4f}\n")

    print("📈 Step 4: Additional Performance Metrics and Learning Curves")

    # Fit and predict to calculate precision, recall, F1, and ROC AUC
    model.fit(X, y)
    y_pred_prob = model.predict_proba(X)  # Get probabilities for ROC curve
    y_pred = model.predict(X)

    print("🔹 Classification Report:\n", classification_report(y, y_pred))
    print("🔹 Confusion Matrix:\n", confusion_matrix(y, y_pred))

    # ROC-AUC curve
    plot_roc_curve(y, y_pred_prob)

    # Plot learning curve
    plot_learning_curve(model, X, y)

    print("\n🎉 [COMPLETED] Supervised Prediction Workflow Finished.\n")


if __name__ == "__main__":
    main()
