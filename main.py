from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = FastAPI()
templates = Jinja2Templates(directory="MLE24Titanic/templates")

# Load the trained model
with open('MLE24Titanic/stacking_clf.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)


# Define label encoders for categorical features
label_encoders = {}
categorical_features = ['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

for feature in categorical_features:
    label_encoders[feature[:-2]] = LabelEncoder()
    label_encoders[feature[:-2]].fit(df[feature])


@app.on_event("startup")
async def startup_event():
    # Fit the label encoders to the respective features
    with open('MLE24Titanic/sex_encoder.pkl', 'rb') as f:
        sex_encoder.fit(pickle.load(f))
    with open('MLE24Titanic/embarked_encoder.pkl', 'rb') as f:
        embarked_encoder.fit(pickle.load(f))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, pclass: int = Form(...), sex_female: int = Form(...), sex_male: int = Form(...), age: int = Form(...), sibsp: int = Form(...),
                  parch: int = Form(...), fare: int = Form(...), embarked_c: int = Form(...), embarked_q: int = Form(...), embarked_s: int = Form(...)):
    # Make prediction
    prediction = stacking_clf.predict([[pclass, sex_female, sex_male, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s]])
    result = "likely" if prediction == 1 else "unlikely"
    
    return templates.TemplateResponse("results.html", {"request": request, "prediction": result})

