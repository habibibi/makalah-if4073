
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


df = pd.read_csv('data.csv')

X = df.drop(columns=['class','s_entropy'])
y = df['class']  

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],          # Banyak tree
    'max_features': ['sqrt', 'log2'],        # Banyak fitur yang dipertimbangkan
    'min_samples_split': [2, 5, 10],         # Sampel minimum untuk split node
    'min_samples_leaf': [1, 2, 4],           # Sampel minimum untuk menjadi node daun
    'max_depth': [None, 10, 20]              # Kedalaman maksimum tree
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',            # Menggungakan akurasi sebagai metrik
    cv=5,                          # 5-fold cross-validation
    verbose=2,                     # Tampilkan proses
    n_jobs=-1                      # Menggunakan seluruh core cpu
)

grid_search.fit(X_train, y_train)

# menampilkan parameter terbaik
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)