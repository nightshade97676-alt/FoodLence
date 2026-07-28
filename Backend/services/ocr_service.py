import base64
import io
import json
import os
import traceback

import google.generativeai as genai
from PIL import Image

LANGUAGES = {"en": "English", "ta": "Tamil", "hi": "Hindi", "es": "Spanish", "fr": "French"}

# Google's Gemini API. As of mid-2026 the current generally-available
# workhorse multimodal model is gemini-3.5-flash (fast, cheap, natively
# multimodal — handles the label photo directly, no separate OCR step).
# gemini-3.1-flash-lite is kept as a fallback in case 3.5 is ever
# rate-limited/unavailable on your key's tier.
# If this ever starts failing, check https://ai.google.dev/gemini-api/docs/models
# for whatever Google's current recommended flash model is and swap it in here.
MODEL_CANDIDATES = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON output in ```json ... ``` even when told not
    to. Strip that off before parsing rather than failing on it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


class OCRService:
    def __init__(self, api_key: str = None):
        # No hardcoded fallback key — a key baked into source is a key that leaks.
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if api_key:
            api_key = api_key.strip()  # guards against a stray copy-paste space/newline
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it as an environment variable "
                "(Render: Dashboard -> your service -> Environment) rather than "
                "hardcoding it in source. Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)
        self.models = MODEL_CANDIDATES

        # Safe diagnostic: confirms what actually got loaded without ever
        # printing the real secret — check your host's logs after a deploy
        # to verify the key isn't truncated or malformed.
        masked = f"{api_key[:4]}...{api_key[-4:]} (length {len(api_key)})" if len(api_key) > 8 else "(too short to mask safely)"
        print(f"[OCRService] Loaded GEMINI_API_KEY: {masked}")

    def analyze_image(self, image_bytes: bytes, user_profile: dict, lang: str = "en") -> dict:
        # Re-encode through PIL to normalize format/orientation before sending.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

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

        image_part = {"mime_type": "image/jpeg", "data": jpeg_bytes}

        errors = []
        for model_name in self.models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [prompt, image_part],
                    generation_config=genai.types.GenerationConfig(
                        # Ask Gemini's native JSON mode for a clean parse;
                        # we still run it through _strip_code_fence as a
                        # belt-and-braces fallback in case a model ignores it.
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                raw = (response.text or "").strip()
                if not raw:
                    finish_reason = getattr(response.candidates[0], "finish_reason", "unknown") if response.candidates else "no candidates"
                    raise ValueError(f"model returned empty content (finish_reason={finish_reason})")
                data = json.loads(_strip_code_fence(raw))
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
