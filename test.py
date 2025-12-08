import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import re
import string
def clean_text(text):
    text = text.lower()

    
    text=re.sub(r"<.*?>", " ", text)

    
    text=re.sub(r"http\S+|www\S+", " ", text)

    
    text=re.sub(r"\S+@\S+", " ", text)

    
    text=re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    
    text=re.sub(r"\d+", " ", text)

    
    text=re.sub(r"\s+", " ", text).strip()

    
    words=[word for word in text.split() if word not in stop_words]
    text= " ".join(words)

    return text
input_text='When shall we three meet againe?In Thunder, Lightning, or in Raine?'
data_cleaned=clean_text(input_text)
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer=tf.keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode='int', 
)
tokenizer.adapt([data_cleaned])
output=tokenizer(data_cleaned)
output=np.array(output)
seq_len=20
pad_output=pad_sequences([output],maxlen=seq_len,padding='pre')
pad_output=np.array(pad_output)
lstm_model=tf.keras.models.load_model('lstm_model.keras')
result=lstm_model.predict([pad_output])
continue_loop=True
import pickle
vocabulary=pickle.load(open('vocabulary.pkl','rb'))
while True:
    index=np.argmax(result)
    word=vocabulary[index]
    print(word)
    result=np.delete(result,index)
    length=len(result)
    user=input("Enter 0 to stop, anynumber to continue: ")
    try:
        user=int(user)
    except ValueError:
        print("Invalid input. Please a number")
            
    if user==0:
        break
    if length==0:
        continue_loop=False


