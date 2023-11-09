from flask import Flask, request, jsonify
import joblib

# Flask 애플리케이션과 모델을 로드합니다.
app = Flask(__name__)
model = joblib.load('data_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# 예측을 수행하는 API 엔드포인트를 정의합니다.
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
