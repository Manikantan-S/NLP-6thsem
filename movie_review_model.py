import nltk
import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import kagglehub
import os
import pickle

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,),
                                initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        return x * K.softmax(e, axis=1)

class MovieReviewAnalyzer:
    def __init__(self):
        self.max_words = 20000
        self.max_length = 200
        self.model = None
        self.tokenizer = None
        self.model_dir = "saved_model"
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Try to load pre-trained model and tokenizer
        self.load_model_and_tokenizer()
        
    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
        return text

    def load_and_prepare_data(self):
        # Download IMDB dataset
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        dataset_path = os.path.join(path, "IMDB Dataset.csv")
        df = pd.read_csv(dataset_path)
        
        # Clean and preprocess data
        df['clean_review'] = df['review'].apply(self.clean_text)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Tokenize text
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(df['clean_review'])
        sequences = self.tokenizer.texts_to_sequences(df['clean_review'])
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length)
        y = np.array(df['sentiment'])
        
        return X, y

    def build_model(self):
        input_layer = Input(shape=(self.max_length,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(input_layer)
        lstm_layer = Bidirectional(LSTM(64))(embedding_layer)
        dense_layer = Dense(64, activation="relu")(lstm_layer)
        output_layer = Dense(1, activation="sigmoid")(dense_layer)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def save_model_and_tokenizer(self):
        # Save model
        self.model.save(os.path.join(self.model_dir, "sentiment_model.h5"))
        
        # Save tokenizer
        with open(os.path.join(self.model_dir, "tokenizer.pickle"), "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model_and_tokenizer(self):
        model_path = os.path.join(self.model_dir, "sentiment_model.h5")
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.pickle")
        
        try:
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                # Load model with custom layer
                self.model = load_model(model_path, custom_objects={'Attention': Attention})
                
                # Load tokenizer
                with open(tokenizer_path, "rb") as handle:
                    self.tokenizer = pickle.load(handle)
                return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        return False

    def train_model(self):
        # Download NLTK data
        nltk.download('punkt')
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train model
        self.build_model()
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)
        
        # Save model and tokenizer
        self.save_model_and_tokenizer()
        
        return self.model.evaluate(X_test, y_test, verbose=0)[1]

    def predict_review(self, review):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not available. Please ensure the model is properly loaded.")
            
        review = self.clean_text(review)
        seq = self.tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=self.max_length)
        pred = self.model.predict(padded, verbose=0)[0][0]
        return "Positive" if pred > 0.5 else "Negative", float(pred)

    def get_rotten_tomatoes_reviews(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            review_elements = soup.find_all('p', class_='review-text')
            if not review_elements:
                review_elements = soup.find_all('div', {'class': 'the_review'})
            
            reviews = []
            for i in range(min(5, len(review_elements))):
                review_text = review_elements[i].get_text().strip()
                reviews.append(review_text)
            
            return reviews if reviews else None
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def analyze_reviews(self, reviews):
        if not reviews:
            return None, None
            
        combined_reviews = " ".join(reviews)
        sentiment, confidence = self.predict_review(combined_reviews)
        
        return sentiment, confidence