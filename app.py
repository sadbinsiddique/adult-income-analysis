from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Adult Income Prediction API",
    description="An API to predict whether an individual's income exceeds $50K/year based on various features.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

@app.get(
    "/", 
    response_class=HTMLResponse
    status_code=200
    summary="Home Page",
    description="Renders the home page of the Adult Income Prediction API."
    tags=["Home"]
)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)