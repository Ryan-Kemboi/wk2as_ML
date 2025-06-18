import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from itertools import cycle

# Simulated unstructured medical notes (for NLP)
notes = [
    "frequent urination and excessive thirst",
    "blurred vision and fatigue",
    "no complaints, healthy",
    "mild headache but no other symptoms",
    "constant hunger and high blood sugar levels",
    "patient feels fine",
    "weight loss and excessive thirst",
    "low energy and skin infections",
    "normal check-up",
    "recurrent yeast infections and blurry vision"
]

# Load structured dataset (Pima Indians Diabetes dataset from Kaggle)
# File: diabetes.csv should be in the same directory
diabetes_data = pd.read_csv("diabetes.csv")

# Add synthetic doctor notes (NLP column)
diabetes_data['DoctorNotes'] = [notes[i % len(notes)] for i in range(len(diabetes_data))]

# Define features and labels
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

text_col = 'DoctorNotes'
numeric_cols = X.columns.drop(text_col)

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer())
])

# Combine numeric and text transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('txt', text_transformer, text_col)
    ]
)

# Final pipeline with Random Forest classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Limit TF-IDF features to avoid high memory usage
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=100))
])

# Optionally: Load smaller subset
diabetes_data = pd.read_csv("diabetes.csv").sample(n=300, random_state=42)