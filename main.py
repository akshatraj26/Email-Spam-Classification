# Spam Email Classification

# Load the Dependencies
import joblib
from fastapi import FastAPI, Depends
from fastapi.encoders import jsonable_encoder
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
    unique_id : str = Field(default_factory=uuid4)
    email: str
    
    
class UserOut(BaseModel):
    result: str
    



# Intialize the FastAPI
app = FastAPI(title="Spam Email Classification", description="This will take the email text gives you whether this is spam or ham email")

# Load the Encoders, Vectorizer and Models
encoders = glob("encoder_vectorizer/**")
models = glob('models/**')

optimal_extra = models[3]
tfidf = encoders[1]
encoder = encoders[0]

model = joblib.load(optimal_extra)
tfidf = joblib.load(encoders[1])
encoder = joblib.load(encoders[0])





# Preprocess the text
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = " ".join(text)
    
    return text



text = "Congratulations! You've won a free ticket to the Bahamas. Click here to claim."
preprocess_text(text)

def make_prediction(clean_text):
    vectorizer = tfidf.transform([clean_text])
    prediction = model.predict(vectorizer)
    label = encoder.inverse_transform(prediction)
    return label[0]


@app.post("/predict", response_model=UserOut)
async def prediction(item: UserIn):
    cleaned_email = preprocess_text(item.email)
    print(cleaned_email)
    label = make_prediction(cleaned_email)
    print(label)
    return UserOut(result=label)
    