# LSTM Text Prediction 

This project implements a Long Short-Term Memory (LSTM) neural network trained on Shakespeare's Macbeth text to predict the next word in a sentence. It includes both a Jupyter notebook for model training and a Flask web application for inference.

## Project Overview

The application trains an LSTM model on cleaned Shakespeare text and provides a web interface to predict the next N words given an input sentence.

---

## Files in This Project

### 1. **lstm.ipynb** (Jupyter Notebook)
The main training pipeline that builds and trains the LSTM model.

**Key Cells:**

- **Cell 1: Load Data**
  - Downloads Shakespeare's Macbeth from NLTK's Gutenberg corpus
  - Loads the raw text data for processing

- **Cell 2: Save Dataset**
  - Writes the raw text to `dataset.txt` for backup

- **Cell 3: Text Cleaning**
  - Implements `clean_text()` function that:
    - Converts text to lowercase
    - Removes HTML tags
    - Removes URLs and email addresses
    - Removes punctuation
    - Removes digits
    - Removes extra whitespace
    - Filters out English stopwords
  - Cleans the entire dataset

- **Cell 4: Tokenization**
  - Uses TensorFlow's `TextVectorization` layer
  - Converts text to integer sequences
  - Vocabulary size: 2000 tokens

- **Cell 5: Create Training Data**
  - Creates sequences of length 20 (X) and their next word (y)
  - Generates training pairs from the tokenized text

- **Cell 6: Get Vocabulary Size**
  - Calculates the vocabulary size (2000 words)
  - Used for model architecture

- **Cell 7: Train-Test Split**
  - Splits data into 80% training and 20% testing
  - Uses random_state=42 for reproducibility

- **Cell 8: Build LSTM Model**
  - Architecture:
    - Embedding layer (128 dimensions)
    - LSTM layer with 64 units (returns sequences)
    - Dropout layer (50% rate)
    - LSTM layer with 64 units
    - Dense output layer (softmax activation)
  - Optimizer: Adam
  - Loss: Sparse Categorical Crossentropy

- **Cell 9: Train Model**
  - Trains for up to 100 epochs
  - Batch size: 32
  - Early stopping with patience=20
  - Monitors validation loss

- **Cell 10: Generate Predictions**
  - Makes predictions on test set

- **Cell 11: Display Predictions**
  - Shows the model output

- **Cell 12: Save Model**
  - Saves trained model as `lstm_model.keras`

- **Cell 13: Save Vocabulary**
  - Saves tokenizer vocabulary as `vocabulary.pkl`
  - Pickle format for easy loading

---

### 2. **app.py** (Flask Backend)
The web server that handles prediction requests.

**Key Functions:**

- **`clean_text(text)`**
  - Same text cleaning function as notebook
  - Preprocesses user input

- **`predict_next_words(input_text, num_words)`**
  - Loads the pre-trained LSTM model
  - Tokenizes input text
  - Pads sequences to length 20
  - Generates predictions
  - Returns top N predicted words by probability

**Routes:**

- **`GET /`**
  - Serves the main web interface (`index.html`)

- **`POST /predict`**
  - Accepts JSON with `text` and `num_words`
  - Returns predicted words as JSON
  - Example: `{"predictions": ["word1", "word2"], "input": "user text"}`

**Startup:**
- Runs on `http://localhost:5000`
- Loads model and vocabulary on startup
- Debug mode enabled with hot-reload disabled

---

### 3. **Supporting Files**

#### **lstm_model.keras**
- Pre-trained LSTM model
- Trained on cleaned Shakespeare text
- Ready for inference

#### **vocabulary.pkl**
- Vocabulary list (2000 words)
- Pickle format
- Indexed mapping: index → word

#### **dataset.txt**
- Original Shakespeare Macbeth text
- Raw data backup

#### **test.py**
- Standalone test script
- Demonstrates model usage without web interface
- Command-line word prediction with interactive loop

#### **requirements.txt**
- Python dependencies:
  - TensorFlow
  - Flask
  - NumPy
  - NLTK
  - Scikit-learn
  - And more

---

## How It Works

### Training Pipeline (Jupyter Notebook)
1. Load Shakespeare text from NLTK
2. Clean text (remove punctuation, stopwords, etc.)
3. Tokenize to integers (max 2000 tokens)
4. Create sequences of 20 words → next word
5. Build and train LSTM model
6. Save model and vocabulary

### Inference Pipeline (Flask App)
1. User enters text in web interface
2. Select number of words to predict (1-10)
3. Flask receives request
4. Cleans input text using same function
5. Tokenizes and pads sequence
6. LSTM predicts probabilities for next word
7. Returns top N predictions
8. Frontend displays results as tags

---

## Usage

### Running the Web App

```bash
python3 app.py
```

Then open browser to: `http://localhost:5000`

### Making Predictions

1. Enter a sentence or phrase
2. Select how many words to predict (1-10)
3. Click "Predict Words"
4. View predicted words displayed as tags

### Example Input
- Input: "When shall we three meet"
- Predicted: "againe", "in", "thunder", "again", "together"

---

## Model Architecture

```
Input (Sequence of 20 words)
    ↓
Embedding (128 dimensions)
    ↓
LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.5)
    ↓
LSTM (64 units)
    ↓
Dense (2000 units, softmax)
    ↓
Output (Probability distribution over vocabulary)
```

---

## Dependencies

- **TensorFlow/Keras**: Neural network framework
- **Flask**: Web server
- **NumPy**: Numerical computing
- **NLTK**: Natural language processing (stopwords, text)
- **Scikit-learn**: Train-test split
- **Pickle**: Vocabulary serialization

---

## Notes

- The model is trained on Shakespeare's Macbeth (~30KB of text)
- Vocabulary limited to 2000 most frequent words
- Sequences padded to fixed length of 20
- Clean text removes stopwords and punctuation
- Web interface supports 1-10 word predictions
- Model runs in CPU-friendly mode with suppressed TensorFlow warnings

---

## Future Improvements

- Train on larger corpus
- Increase vocabulary size
- Implement beam search for better predictions
- Add model evaluation metrics display
- Support multiple languages
- Fine-tuning parameters based on performance
