import nltk
import string
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, jsonify
import os

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

app = Flask(__name__)

try:
    lstm_model = tf.keras.models.load_model('lstm_model.keras')
    print("✓ LSTM model loaded successfully")
except Exception as e:
    print(f"✗ Error loading LSTM model: {e}")
    lstm_model = None

vocabulary = []
try:
    with open('vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    print(f"✓ Vocabulary loaded successfully ({len(vocabulary)} words)")
    print(f"  Vocabulary type: {type(vocabulary).__name__}")
except Exception as e:
    print(f"✗ Error loading vocabulary: {e}")

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    
    text = re.sub(r"<.*?>", " ", text)
    
    
    text = re.sub(r"http\S+|www\S+", " ", text)
    
    
    text = re.sub(r"\S+@\S+", " ", text)
    
    
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    
    text = re.sub(r"\d+", " ", text)
    
    
    text = re.sub(r"\s+", " ", text).strip()
    
    
    words = [word for word in text.split() if word not in stop_words]
    text = " ".join(words)
    
    return text

def predict_next_words(input_text, num_words=1):
    """Predict next words using the LSTM model"""
    try:
        if not lstm_model:
            return ["Error: LSTM model not loaded"]
        if not vocabulary:
            return ["Error: Vocabulary not loaded"]
        
        
        cleaned_text = clean_text(input_text)
        
        if not cleaned_text:
            return ["Error: Text is empty after cleaning"]
        
        
        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=2000,
            output_mode='int',
        )
        tokenizer.adapt([cleaned_text])
        
        
        output = tokenizer(cleaned_text)
        output = np.array(output)
        
        
        seq_len = 20
        pad_output = pad_sequences([output], maxlen=seq_len, padding='pre')
        pad_output = np.array(pad_output)
        
        
        result = lstm_model.predict(pad_output, verbose=0)[0]
        predicted_words = []
        
        
        for i in range(num_words):
            if len(result) == 0:
                break
            index = int(np.argmax(result))
            
            
            if 0 <= index < len(vocabulary):
                word = vocabulary[index]
                if word:
                    predicted_words.append(str(word).strip())
            
            result = np.delete(result, index)
        
        if not predicted_words:
            return ["No prediction available"]
        
        print(f"✓ Predictions: {predicted_words}")
        return predicted_words
    except Exception as e:
        print(f"✗ Error in predict_next_words: {str(e)}")
        import traceback
        traceback.print_exc()
        return [f"Error: {str(e)}"]

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for text prediction"""
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip() if data else ''
        num_words = int(data.get('num_words', 1)) if data else 1
        
        
        if num_words < 1 or num_words > 20:
            num_words = 1
        
        print(f"Received request: text='{input_text[:50]}...', num_words={num_words}")
        
        if not input_text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        predicted_words = predict_next_words(input_text, num_words)
        
        print(f"Returning: predictions={predicted_words}, input={input_text[:50]}...")
        
        response_data = {
            'predictions': predicted_words if isinstance(predicted_words, list) else [str(predicted_words)],
            'input': input_text
        }
        
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("LSTM Text Prediction Web App")
    print("="*50)
    print("Starting Flask server on http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
