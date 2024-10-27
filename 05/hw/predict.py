import os
import pickle

from flask import Flask, request, jsonify

model_fname = os.getenv('MODEL_FILE_NAME', 'model1.bin')
print(model_fname)

with open(model_fname, 'rb') as f_in:
    model = pickle.load(f_in)
print("load model")

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'subscription_probability': float(y_pred),
        'subscription': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
