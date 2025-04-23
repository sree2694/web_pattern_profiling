import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === 1. Load Data ===
from data_simulation import generate_fake_data as generate_user_data

def load_data():
    df = generate_user_data(num_users=1000)
    df['sequence_str'] = df['browsing_sequence'].apply(lambda seq: ' '.join(seq))
    return df

# === 2. Preprocessing ===
def preprocess_data(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['sequence_str'])

    le = LabelEncoder()
    y = le.fit_transform(df['gender'])

    return X, y, vectorizer, le

# === 3. Train Model ===
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[Classification Report]:")
    print(classification_report(y_test, y_pred))

    return model

# === 4. Main Function ===
def main():
    print("[+] Loading data and training supervised gender prediction model...")
    df = load_data()
    X, y, _, _ = preprocess_data(df)
    model = train_model(X, y)
    print("[✔] Model training completed.")

if __name__ == "__main__":
    main()
