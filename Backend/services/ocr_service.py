import io
import json
import os
import traceback

from google import genai
from google.genai import types
from PIL import Image


LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
}


# Current Gemini model
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
]


def _strip_code_fence(text: str) -> str:
    """
    Removes ```json ... ``` if Gemini returns JSON
    inside a markdown code block.
    """

    text = text.strip()

    if text.startswith("```"):
        if "\n" in text:
            text = text.split("\n", 1)[1]
        else:
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    return text


class OCRService:

    def __init__(self, api_key: str = None):

        # ---------------------------------------------------------
        # GET GEMINI API KEY
        # ---------------------------------------------------------

        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if api_key:
            api_key = api_key.strip()

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add GEMINI_API_KEY in Render Environment Variables."
            )

        # ---------------------------------------------------------
        # CREATE GEMINI CLIENT
        # ---------------------------------------------------------

        self.client = genai.Client(
            api_key=api_key
        )

        self.models = MODEL_CANDIDATES

        # Safe diagnostic.
        # NEVER print the complete API key.
        if len(api_key) > 8:
            masked = (
                f"{api_key[:4]}..."
                f"{api_key[-4:]}"
                f" (length {len(api_key)})"
            )
        else:
            masked = "(key too short)"

        print(
            f"[OCRService] GEMINI_API_KEY loaded: {masked}"
        )

    # =========================================================
    # ANALYZE IMAGE
    # =========================================================

    def analyze_image(
        self,
        image_bytes: bytes,
        user_profile: dict,
        lang: str = "en"
    ) -> dict:

        # ---------------------------------------------------------
        # 1. OPEN IMAGE
        # ---------------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # ---------------------------------------------------------
        # 2. CONVERT IMAGE TO JPEG
        # ---------------------------------------------------------

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=95
        )

        jpeg_bytes = buffer.getvalue()

        # ---------------------------------------------------------
        # 3. LANGUAGE
        # ---------------------------------------------------------

        target_lang = LANGUAGES.get(
            lang,
            "English"
        )

        # ---------------------------------------------------------
        # 4. USER RESTRICTIONS
        # ---------------------------------------------------------

        active_restrictions = []

        for key, value in user_profile.items():

            if (
                value is True
                and key not in ("age", "weight")
            ):
                active_restrictions.append(
                    key.replace("_", " ").title()
                )

        if active_restrictions:
            restrictions_str = ", ".join(
                active_restrictions
            )
        else:
            restrictions_str = (
                "None "
                "(general health assessment only)"
            )

        # ---------------------------------------------------------
        # 5. GEMINI PROMPT
        # ---------------------------------------------------------

        prompt = f"""
You are an OCR and ingredient-extraction assistant
for a food label scanner called Food Lens.

Look carefully at the attached food label image.

Read the product name and every ingredient that can
actually be seen on the label.

USER'S SELECTED HEALTH/DIETARY RESTRICTIONS:

{restrictions_str}

The user wants the final display language to be:

{target_lang}

Return ONLY valid JSON.

Do NOT return markdown.

Do NOT return ```json.

Do NOT add explanations outside the JSON.

Use EXACTLY this JSON structure:

{{
    "product_name": "product name in English",

    "product_name_translated":
        "product name translated into {target_lang}",

    "detected_ingredients": [
        "ingredient 1 in English",
        "ingredient 2 in English"
    ],

    "detected_ingredients_translated": [
        "ingredient 1 in {target_lang}",
        "ingredient 2 in {target_lang}"
    ],

    "safer_alternatives": [
        "alternative product 1 in {target_lang}",
        "alternative product 2 in {target_lang}",
        "alternative product 3 in {target_lang}"
    ]
}}

IMPORTANT RULES:

1. "detected_ingredients" MUST be in English.

2. The English ingredient list is used by the
   Food Lens automated rules engine.

3. "detected_ingredients_translated" must contain
   the SAME ingredients translated into {target_lang}.

4. Do not invent ingredients.

5. If an ingredient cannot be read clearly,
   do not guess it.

6. If the image is completely unreadable,
   return empty ingredient lists.

7. Suggest 2-3 realistic healthier packaged-food
   alternatives.

8. The alternatives must be written in {target_lang}.

9. The product name should remain in English in
   "product_name".

10. The translated product name must be in
    {target_lang}.
"""

        # ---------------------------------------------------------
        # 6. IMAGE FOR GEMINI
        # ---------------------------------------------------------

        image_part = types.Part.from_bytes(
            data=jpeg_bytes,
            mime_type="image/jpeg"
        )

        # ---------------------------------------------------------
        # 7. CALL GEMINI
        # ---------------------------------------------------------

        errors = []

        for model_name in self.models:

            try:

                print(
                    f"[OCRService] Calling Gemini model: "
                    f"{model_name}"
                )

                response = self.client.models.generate_content(
                    model=model_name,

                    contents=[
                        prompt,
                        image_part
                    ],

                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )

                # -------------------------------------------------
                # 8. GET RESPONSE TEXT
                # -------------------------------------------------

                raw = (
                    response.text or ""
                ).strip()

                if not raw:

                    finish_reason = "unknown"

                    try:
                        if response.candidates:
                            finish_reason = (
                                response
                                .candidates[0]
                                .finish_reason
                            )
                    except Exception:
                        pass

                    raise ValueError(
                        "Gemini returned empty content. "
                        f"Finish reason: {finish_reason}"
                    )

                print(
                    f"[OCRService] Gemini response received "
                    f"from {model_name}"
                )

                # -------------------------------------------------
                # 9. CLEAN JSON
                # -------------------------------------------------

                cleaned = _strip_code_fence(
                    raw
                )

                # -------------------------------------------------
                # 10. PARSE JSON
                # -------------------------------------------------

                data = json.loads(
                    cleaned
                )

                # -------------------------------------------------
                # 11. NORMALIZE
                # -------------------------------------------------

                return self._normalize(
                    data
                )

            except Exception as exc:

                error_message = (
                    f"{model_name}: {exc}"
                )

                errors.append(
                    error_message
                )

                print(
                    f"[OCRService] Gemini model "
                    f"'{model_name}' failed:"
                )

                print(
                    traceback.format_exc()
                )

                # Try next model if one exists.
                continue

        # ---------------------------------------------------------
        # 12. ALL MODELS FAILED
        # ---------------------------------------------------------

        print(
            "[OCRService] All Gemini models failed."
        )

        for error in errors:
            print(
                f"[OCRService] {error}"
            )

        return self._fallback(
            errors
        )

    # =========================================================
    # NORMALIZE GEMINI RESULT
    # =========================================================

    @staticmethod
    def _normalize(
        data: dict
    ) -> dict:

        data.setdefault(
            "product_name",
            "Scanned Product"
        )

        data.setdefault(
            "product_name_translated",
            data["product_name"]
        )

        data.setdefault(
            "detected_ingredients",
            []
        )

        data.setdefault(
            "detected_ingredients_translated",
            data["detected_ingredients"]
        )

        data.setdefault(
            "safer_alternatives",
            []
        )

        return data

    # =========================================================
    # FALLBACK
    # =========================================================

    @staticmethod
    def _fallback(
        errors: list
    ) -> dict:

        if errors:
            detail = errors[-1]
        else:
            detail = "Unknown Gemini API error"

        return {
            "product_name":
                "Scanned Food Product (offline fallback)",

            "product_name_translated":
                "Scanned Food Product (offline fallback)",

            "detected_ingredients": [
                "Water",
                "Sugar",
                "Wheat Flour",
                "Milk Powder",
                "Salt"
            ],

            "detected_ingredients_translated": [
                "Water",
                "Sugar",
                "Wheat Flour",
                "Milk Powder",
                "Salt"
            ],

            "safer_alternatives": [
                "Gemini API call failed. "
                f"Last error: {detail}"
            ]
        }
