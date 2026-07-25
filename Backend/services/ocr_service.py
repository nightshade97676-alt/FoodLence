import base64
import io
import json
import os
import traceback

from openai import OpenAI
from PIL import Image

LANGUAGES = {"en": "English", "ta": "Tamil", "hi": "Hindi", "es": "Spanish", "fr": "French"}

# Groq's OpenAI-compatible endpoint. Groq's vision-model lineup changes
# often — meta-llama/llama-4-scout-17b-16e-instruct and
# meta-llama/llama-4-maverick-17b-128e-instruct (the two models most
# tutorials reference) were BOTH deprecated by Groq earlier in 2026.
# qwen/qwen3.6-27b is the current vision-capable model as of mid-2026.
# If this ever starts failing, check console.groq.com/docs/model for
# whatever Groq's current vision model is and swap it in here.
MODEL_CANDIDATES = ["qwen/qwen3.6-27b"]


class OCRService:
    def __init__(self, api_key: str = None):
        # No hardcoded fallback key — same reasoning as every other provider
        # in this project: a key baked into source is a key that leaks.
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key:
            api_key = api_key.strip()  # guards against a stray copy-paste space/newline
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it as an environment variable "
                "(Render: Dashboard -> your service -> Environment) rather than "
                "hardcoding it in source. Get a free key at https://console.groq.com/keys"
            )
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=30.0)
        self.models = MODEL_CANDIDATES

        # Safe diagnostic: confirms what actually got loaded without ever
        # printing the real secret — check your host's logs after a deploy
        # to verify the key isn't truncated or malformed.
        masked = f"{api_key[:4]}...{api_key[-4:]} (length {len(api_key)})" if len(api_key) > 8 else "(too short to mask safely)"
        print(f"[OCRService] Loaded GROQ_API_KEY: {masked}")

    def analyze_image(self, image_bytes: bytes, user_profile: dict, lang: str = "en") -> dict:
        # Re-encode through PIL to normalize format/orientation before sending.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

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
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                data = json.loads(raw)
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
                f"Groq API call failed, showing placeholder data. Last error: {detail}",
            ],
        }
