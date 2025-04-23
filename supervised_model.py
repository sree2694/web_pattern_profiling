import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_simulation import generate_fake_data
from feature_engineering import extract_features

def main():
    print("\n🔍 [MODE] Supervised Prediction Started")

    print("\n📦 Step 1: Generating user data...")
    df = generate_fake_data(num_users=100)
    print("✅ User data generated.\n")

    print("🧪 Step 2: Extracting features...")
    df = extract_features(df)
    print("✅ Features extracted: ['sequence_length', 'unique_domains', 'repeats']\n")

    X = df[['sequence_length', 'unique_domains', 'repeats']]
    y = df['age_group']  # Or switch to 'gender'

    print("📊 Step 3: Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"✅ Cross-validation completed. Accuracy: {scores.mean():.4f} ± {scores.std():.4f}\n")

    print("🧠 Step 4: Training and testing the final model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("✅ Model training completed.\n")

    print("📈 Step 5: Evaluation Results")
    print("----------------------------")
    print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
    print("🔹 Classification Report:\n", classification_report(y_test, y_pred))
    print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n🎉 [COMPLETED] Supervised Prediction Workflow Finished.\n")

if __name__ == "__main__":
    main()
