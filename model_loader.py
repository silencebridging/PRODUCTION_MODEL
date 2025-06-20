import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load("model/mlp_tsl_static.pkl")
print("Model successfully loaded from model/mlp_tsl_static.pkl")

le = LabelEncoder()
le.fit([chr(i) for i in range(ord('A'), ord('Z') + 1)])

def predict_letter(X):
    pred_index = model.predict(X)[0]
    return le.inverse_transform([pred_index])[0]
