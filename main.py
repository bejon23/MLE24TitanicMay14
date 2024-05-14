from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import os

app = FastAPI()
templates = Jinja2Templates(directory="MLE24Titanic/templates")

# Load the trained model
with open('stacking_clf.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)

# Load CSS style
def load_css(filename):
    with open(os.path.join("MLE24Titanic/static", filename), "r") as f:
        return f.read()

css_style = load_css("style.css")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "style": css_style})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, pclass: int = Form(...), sex: str = Form(...), age: int = Form(...), sibsp: int = Form(...),
                   parch: int = Form(...), fare: int = Form(...), embarked: str = Form(...)):
    # Make prediction
    prediction = stacking_clf.predict([[pclass, sex, age, sibsp, parch, fare, embarked]])
    result = "likely" if prediction == 1 else "unlikely"
    
    return templates.TemplateResponse("results.html", {"request": request, "style": css_style, "prediction": result})
