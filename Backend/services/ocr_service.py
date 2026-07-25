import io
import json
import os
import traceback

from google import genai
from google.genai import types
from PIL import Image

LANGUAGES = {"en": "English", "ta": "Tamil", "hi": "Hindi", "es": "Spanish", "fr": "French"}

# NOTE: hardcoding an API key in source is not recommended for anything beyond
# local testing — set the GEMINI_API_KEY environment variable instead so the
# key never ends up committed to source control or shared files.
DEFAULT_API_KEY = "AQ.Ab8RN6Lp8zSILnpQXojcZnUPubNqk6HAlCUApB3tNj_iq40DXA"

# As of mid-2026 all gemini-1.5-* and gemini-2.0-* model IDs have been shut
# down by Google (they now return 404). "gemini-flash-latest" is a stable
# alias that always points at Google's current default Flash model, so it's
# the safest first choice — with pinned versions as fallbacks in case the
# alias itself is ever retired.
MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash"]


class OCRService:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY)
        # 20s request timeout so a bad connection can't hang a worker forever.
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=20_000))
        self.models = MODEL_CANDIDATES

    def analyze_with_gemini(self, image_bytes: bytes, user_profile: dict, lang: str = "en") -> dict:
        # Re-encode through PIL to normalize format/orientation before sending.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        target_lang = LANGUAGES.get(lang, "English")
        active = [k.replace("_", " ").title() for k, v in user_profile.items()
                  if v is True and k not in ("age", "weight")]
        restrictions_str = ", ".join(active) if active else "None (general health assessment only)"

        prompt = f"""
You are an OCR and ingredient-extraction assistant for a food label scanner.
Look at the attached food label photo and read every ingredient listed.

USER'S SELECTED HEALTH/DIETARY RESTRICTIONS: {restrictions_str}

Return ONLY valid JSON (no markdown fences, no commentary) with this exact schema:
{{
  "product_name": "product name in English",
  "product_name_translated": "product name translated into {target_lang}",
  "detected_ingredients": ["ingredient 1 in English", "ingredient 2 in English"],
  "detected_ingredients_translated": ["ingredient 1 in {target_lang}", "ingredient 2 in {target_lang}"],
  "safer_alternatives": ["alternative product 1 in {target_lang}", "alternative product 2 in {target_lang}"]
}}

Rules:
- "detected_ingredients" MUST stay in English (used for automated rule-matching).
- "detected_ingredients_translated" is the same list translated into {target_lang} for display.
- If no restrictions were selected, still extract ingredients normally.
- Suggest 2-3 realistic, healthier packaged-food alternatives for "safer_alternatives".
- If the image is unreadable, return empty lists rather than guessing.
"""

        errors = []
        for model_name in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                data = json.loads(response.text.strip())
                return self._normalize(data)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_name}: {exc}")
                print(f"[OCRService] model '{model_name}' failed:\n{traceback.format_exc()}")
                continue

        print(f"[OCRService] all models failed:\n" + "\n".join(errors))
        return self._fallback(errors)

    @staticmethod
    def _normalize(data: dict) -> dict:
        data.setdefault("product_name", "Scanned Product")
        data.setdefault("product_name_translated", data["product_name"])
        data.setdefault("detected_ingredients", [])
        data.setdefault("detected_ingredients_translated", data["detected_ingredients"])
        data.setdefault("safer_alternatives", [])
        return data

    @staticmethod
    def _fallback(errors: list) -> dict:
        # Surface the *real* error in the response itself (not just the
        # server console) so a failure is diagnosable straight from the UI.
        detail = errors[-1] if errors else "Unknown error"
        return {
            "product_name": "Scanned Food Product (offline fallback)",
            "product_name_translated": "Scanned Food Product (offline fallback)",
            "detected_ingredients": ["Water", "Sugar", "Wheat Flour", "Milk Powder", "Salt"],
            "detected_ingredients_translated": ["Water", "Sugar", "Wheat Flour", "Milk Powder", "Salt"],
            "safer_alternatives": [
                f"Gemini API call failed, showing placeholder data. Last error: {detail}",
            ],
        }
