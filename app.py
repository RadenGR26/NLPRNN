from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load tokenizer dan label encoder
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load semua model
models = {
    'RNN': load_model('models/rnn_model.h5'),
    'LSTM': load_model('models/lstm_model.h5'),
    'GRU': load_model('models/gru_model.h5'),
    'BiRNN': load_model('models/birnn_model.h5')
}

# Fungsi prediksi
def predict_sentiment(text, model_type):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    model = models.get(model_type)
    if model:
        pred = model.predict(padded)[0]
        label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        return label, float(np.max(pred))
    return "Model tidak ditemukan", 0.0

# Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        model_type = request.form['model']
        label, confidence = predict_sentiment(text, model_type)
        result = {
            'text': text,
            'model': model_type,
            'label': label,
            'confidence': round(confidence * 100, 2)
        }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
