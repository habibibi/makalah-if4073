
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('data.csv')

X = df.drop(columns=['class','s_entropy'])
y = df['class']  

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    random_state=42, 
    max_features='sqrt',
    min_samples_leaf=1, 
    min_samples_split=2, 
    max_depth=20,
    n_estimators=100)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluasi model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Simpan scaler dan model
with open('model/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)
with open('model/model.pkl','wb') as f:
    pickle.dump(rf,f)