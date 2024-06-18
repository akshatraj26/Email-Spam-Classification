# Spam Email Classification
# This time we have used dependencies injection
# Load the Dependencies
import joblib
from fastapi import FastAPI, Depends
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from pydantic import BaseModel, Field
from typing import Annotated
import os
from glob import glob
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from uuid import UUID, uuid4


# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')


class UserIn(BaseModel):
    unique_id : str = Field(default_factory=lambda: str(uuid4()))
    email: str
    
    
class UserOut(BaseModel):
    result: str
    



# Intialize the FastAPI
app = FastAPI(title="Spam Email Classification", description="This will take the email text gives you whether this is spam or ham email")

# Load the Encoders, Vectorizer and Models
encoders = glob("encoder_vectorizer/**")
models = glob('models/**')

def get_encoder() -> LabelEncoder:
   # This will take saved encoder with labels and load the encoder using joblib
   encoder_path = "encoder_vectorizer/label_encoder.joblib"
   encoder = joblib.load(encoder_path)
   return encoder

def get_vectorizer() -> TfidfVectorizer:
    vectorizer_path = "encoder_vectorizer/tfidf.joblib"
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer

def get_model() -> BaseEstimator:
    model_path = "models/best_spam_email_extratree_98.joblib"
    model = joblib.load(model_path)
    return model

    

# Preprocess the text
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    tokens = [ps.stem(word) for word in text if not word in stop_words]
    clean_text = " ".join(tokens)
    return clean_text

# Dependencies for preprocessed text
def get_preprocessed_text(user_in: UserIn) -> str:
    clean_text = preprocess_text(user_in.email)
    return clean_text



@app.post("/")
async def predict(
    items: UserIn,
    model = Depends(get_model),
    vectorizer = Depends(get_vectorizer),
    encoder = Depends(get_encoder)
    ):
    clean_text = get_preprocessed_text(items)
    features = vectorizer.transform([clean_text])
    prediction = model.predict(features)
    label = encoder.inverse_transform(prediction)
    print("prediction:- ",prediction[0])
    print("Label:- ", label[0])
    response_data =  {'Prediction' : int(prediction[0]), 'label': label[0]}
    return response_data
    