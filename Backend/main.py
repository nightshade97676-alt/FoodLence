"""
Food Lens backend.

Run this file directly with:
    python main.py
from inside the Backend/ folder — see README / chat instructions for the
full setup steps (venv + pip install + run).
"""
import asyncio
import os
import sys

# Make sure this script's own directory is importable regardless of the
# working directory it's launched from (fixes ModuleNotFoundError for
# "services" / "rules" when run via `python -m Backend.main`, `uvicorn`, etc.)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from services.ocr_service import OCRService
from rules.engine import RulesEngine

app = FastAPI(title="Food Lens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # no cookies/auth used — wildcard origin + credentials is invalid per CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_service = OCRService()
rules_engine = RulesEngine()

FRONTEND_PATH = os.path.join(THIS_DIR, "static", "index.html")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    try:
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>Frontend not found</h1><p>Expected Frontend/html/index.html next to Backend/.</p>",
            status_code=500,
        )


@app.post("/analyze-food")
async def analyze_food(
    image: UploadFile = File(...),
    lang: str = Form("en"),
    age: int = Form(0),
    weight: float = Form(0),
    # Dietary preferences
    vegan: bool = Form(False), vegetarian: bool = Form(False), keto: bool = Form(False),
    paleo: bool = Form(False), halal: bool = Form(False), kosher: bool = Form(False),
    gluten_free: bool = Form(False), dairy_free: bool = Form(False),
    # Allergies & intolerances
    peanuts: bool = Form(False), tree_nuts: bool = Form(False), milk_dairy: bool = Form(False),
    eggs: bool = Form(False), soy: bool = Form(False), fish: bool = Form(False),
    shellfish: bool = Form(False), sesame: bool = Form(False), wheat: bool = Form(False),
    mustard: bool = Form(False),
    # Health conditions
    diabetic: bool = Form(False), high_blood_pressure: bool = Form(False),
    lactose_intolerant: bool = Form(False), celiac_disease: bool = Form(False),
    high_cholesterol: bool = Form(False), kidney_care: bool = Form(False),
):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            return JSONResponse({"error": "Empty image upload."}, status_code=400)

        user_profile = {
            "age": age, "weight": weight,
            "vegan": vegan, "vegetarian": vegetarian, "keto": keto, "paleo": paleo,
            "halal": halal, "kosher": kosher, "gluten_free": gluten_free, "dairy_free": dairy_free,
            "peanuts": peanuts, "tree_nuts": tree_nuts, "milk_dairy": milk_dairy, "eggs": eggs,
            "soy": soy, "fish": fish, "shellfish": shellfish, "sesame": sesame, "wheat": wheat,
            "mustard": mustard, "diabetic": diabetic, "high_blood_pressure": high_blood_pressure,
            "lactose_intolerant": lactose_intolerant, "celiac_disease": celiac_disease,
            "high_cholesterol": high_cholesterol, "kidney_care": kidney_care,
        }

        # analyze_with_gemini is a blocking (synchronous) network call — run it
        # in a worker thread so it doesn't stall the whole event loop.
        ocr_result = await asyncio.to_thread(ocr_service.analyze_with_gemini, image_bytes, user_profile, lang)
        score_result = rules_engine.analyze(ocr_result["detected_ingredients"], user_profile)

        return {
            "product_name": ocr_result["product_name_translated"],
            "detected_ingredients": ocr_result["detected_ingredients_translated"],
            "safer_alternatives": ocr_result["safer_alternatives"],
            "safety_percentage": score_result["safety_percentage"],
            "classification": score_result["classification"],
            "nutri_grade": score_result["nutri_grade"],
            "recommendations": score_result["recommendations"],
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
