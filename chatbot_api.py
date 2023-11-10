from flask import Flask, request, jsonify
import joblib

# Flask 애플리케이션과 모델, 벡터라이저, 레이블 인코더를 로드합니다.
app = Flask(__name__)
model = joblib.load('data_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
encoder = joblib.load('encoder.joblib')  # 레이블 인코더 추가

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    prediction_label = encoder.inverse_transform(prediction)  # 숫자를 레이블로 변환
    return jsonify({'prediction': prediction_label[0]})

if __name__ == '__main__':
    app.run(debug=True)
