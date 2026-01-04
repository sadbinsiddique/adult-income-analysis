"""
Adult Income Prediction API
FastAPI application for income prediction using LinearSVC model
"""
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from src.helper import DataHelper, LogHelper, InputDataHelper
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging


# Initialize FastAPI app
app = FastAPI(
    title="Adult Income Prediction API",
    description="An API to predict whether an individual's income exceeds $50K/year based on various features.",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Load model at startup
try:
    predict_pipeline.load_model()
    logging.info("Model loaded successfully at startup")
except Exception as e:
    logging.warning(f"Model not loaded at startup: {str(e)}")


# Pydantic models for API
class PredictionInput(BaseModel):
    age: float
    sex: str
    occupation: str
    education: str
    education_num: float
    native_country: str


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence_score: float


# ============== HTML Routes ==============

@app.get("/", response_class=HTMLResponse, tags=["Pages"])
async def home(request: Request):
    """Render home page"""
    logging.info("Home page accessed")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predictdata", response_class=HTMLResponse, tags=["Pages"])
async def predict_page(request: Request):
    """Render prediction form page"""
    logging.info("Prediction page accessed")
    form_options = InputDataHelper.get_form_options()
    return templates.TemplateResponse(
        "predictdata.html", 
        {
            "request": request,
            "form_options": form_options
        }
    )


@app.post("/predictdata", response_class=HTMLResponse, tags=["Pages"])
async def predict_form(
    request: Request,
    age: float = Form(...),
    sex: str = Form(...),
    occupation: str = Form(...),
    education: str = Form(...),
    education_num: float = Form(...),
    native_country: str = Form(...)
):
    """Handle prediction form submission"""
    logging.info("Prediction form submitted")
    
    try:
        # Create input data (only required fields, others use defaults)
        form_data = {
            'age': age,
            'sex': sex,
            'occupation': occupation,
            'education': education,
            'education_num': education_num,
            'native_country': native_country
        }
        
        logging.info(f"Form data received: {form_data}")
        
        # Create properly encoded DataFrame from form data
        input_df = InputDataHelper.create_input_dataframe(form_data)
        
        # Make prediction directly (input is already encoded)
        prediction = predict_pipeline.predict(input_df)[0]
        confidence = predict_pipeline.predict_proba(input_df)[0]
        
        is_high_income = prediction == 1
        prediction_label = 'Greater than $50,000' if is_high_income else 'Less than or equal to $50,000'
        confidence_percent = abs(confidence) * 10  # Scale for display
        
        # Calculate salary based on prediction
        if is_high_income:
            # >50K: 50000 * confidence
            estimated_salary = 50000 * (confidence_percent / 100)
            salary_formula = f"50000 × {confidence_percent:.2f}%"
        else:
            # <=50K: (50000 / 100) * confidence
            estimated_salary = (50000 / 100) * (confidence_percent / 100)
            salary_formula = f"(50000 / 100) × {confidence_percent:.2f}%"
        
        logging.info(f"Prediction: {prediction_label}, Confidence: {confidence_percent:.2f}%")
        logging.info(f"Estimated Salary: ${estimated_salary:.2f}")
        
        form_options = InputDataHelper.get_form_options()
        
        return templates.TemplateResponse(
            "predictdata.html",
            {
                "request": request,
                "results": True,
                "results_predicted": prediction_label,
                "results_accuracy": min(confidence_percent, 99.9),
                "is_high_income": is_high_income,
                "estimated_salary": estimated_salary,
                "salary_formula": salary_formula,
                "form_options": form_options,
                "form_data": form_data
            }
        )
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        form_options = InputDataHelper.get_form_options()
        return templates.TemplateResponse(
            "predictdata.html",
            {
                "request": request,
                "error": str(e),
                "form_options": form_options
            }
        )


@app.get("/about", response_class=HTMLResponse, tags=["Pages"])
async def about(request: Request):
    """Render about page"""
    logging.info("About page accessed")
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse, tags=["Pages"])
async def contact(request: Request):
    """Render contact page"""
    logging.info("Contact page accessed")
    return templates.TemplateResponse("contact.html", {"request": request})


# ============== API Routes ==============

@app.post("/api/predict", response_model=PredictionResponse, tags=["API"])
async def api_predict(input_data: PredictionInput):
    """
    API endpoint for income prediction
    
    Returns prediction (0 = <=50K, 1 = >50K) with confidence score
    """
    logging.info("API prediction request received")
    
    try:
        # Convert to dict (only required fields)
        form_data = {
            'age': input_data.age,
            'sex': input_data.sex,
            'occupation': input_data.occupation,
            'education': input_data.education,
            'education_num': input_data.education_num,
            'native_country': input_data.native_country
        }
        
        # Create properly encoded DataFrame
        input_df = InputDataHelper.create_input_dataframe(form_data)
        
        # Predict directly (input is already encoded)
        prediction = int(predict_pipeline.predict(input_df)[0])
        confidence = float(predict_pipeline.predict_proba(input_df)[0])
        
        prediction_label = 'Greater than 50K' if prediction == 1 else 'Less than or equal to 50K'
        
        logging.info(f"API Prediction: {prediction_label}")
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence_score=abs(confidence)
        )
        
    except Exception as e:
        logging.error(f"API prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs", tags=["API"])
async def get_logs(lines: int = 50):
    """
    Get recent log entries
    
    Args:
        lines: Number of log lines to return (default 50)
    """
    logs_data = LogHelper.read_logs(num_lines=lines)
    return JSONResponse(content=logs_data)

@app.get("/api/health", tags=["API"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predict_pipeline.model is not None,
        "version": "1.0.0"
    }


@app.get("/api/form-options", tags=["API"])
async def get_form_options():
    """Get all form dropdown options"""
    return InputDataHelper.get_form_options()


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server")
    uvicorn.run(app, host="127.0.0.1", port=8000)
