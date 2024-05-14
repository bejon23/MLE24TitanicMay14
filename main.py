from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import os

app = FastAPI()

import os

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the templates directory
templates_dir = os.path.join(current_dir, "templates")

# Initialize Jinja2Templates with the correct directory
templates = Jinja2Templates(directory=templates_dir)


# Load the trained model
with open('stacking_clf.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, pclass: int = Form(...), sex: str = Form(...), age: int = Form(...), sibsp: int = Form(...),
                   parch: int = Form(...), fare: int = Form(...), embarked: str = Form(...)):
    # Make prediction
    prediction = stacking_clf.predict([[pclass, sex, age, sibsp, parch, fare, embarked]])
    result = "likely" if prediction == 1 else "unlikely"
    
    return templates.TemplateResponse("results.html", {"request": request, "prediction": result})
