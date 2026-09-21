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


MODEL_CANDIDATES = [
    "gemini-3.8-flash",
]


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON output in markdown code fences."""

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
                "Set it as an environment variable "
                "(Render: Dashboard -> your service -> Environment)."
            )

        # ---------------------------------------------------------
        # CREATE NEW GEMINI CLIENT
        # ---------------------------------------------------------

        self.client = genai.Client(
            api_key=api_key
        )

        self.models = MODEL_CANDIDATES

        # Safe diagnostic
        masked = (
            f"{api_key[:4]}...{api_key[-4:]} "
            f"(length {len(api_key)})"
            if len(api_key) > 8
            else "(too short to mask safely)"
        )

        print(
            f"[OCRService] Loaded GEMINI_API_KEY: {masked}"
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

        buf = io.BytesIO()

        image.save(
            buf,
            format="JPEG"
        )

        jpeg_bytes = buf.getvalue()

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

        active = [
            k.replace("_", " ").title()
            for k, v in user_profile.items()
            if v is True and k not in ("age", "weight")
        ]

        restrictions_str = (
            ", ".join(active)
            if active
            else "None (general health assessment only)"
        )

        # ---------------------------------------------------------
        # 5. GEMINI PROMPT
        # ---------------------------------------------------------

        prompt = f"""
You are an OCR and ingredient-extraction assistant for a food label scanner.

Look at the attached food label photo and read every ingredient listed.

USER'S SELECTED HEALTH/DIETARY RESTRICTIONS:
{restrictions_str}

Return ONLY valid JSON (no markdown fences, no commentary) with this exact schema:

{{
    "product_name": "product name in English",

    "product_name_translated":
        "product name translated into {target_lang}",

    "detected_ingredients":
        ["ingredient 1 in English", "ingredient 2 in English"],

    "detected_ingredients_translated":
        ["ingredient 1 in {target_lang}", "ingredient 2 in {target_lang}"],

    "safer_alternatives":
        [
            "alternative product 1 in {target_lang}",
            "alternative product 2 in {target_lang}"
        ]
}}

Rules:

- "detected_ingredients" MUST stay in English (used for automated rule-matching).
- "detected_ingredients_translated" is the same list translated into {target_lang} for display.
- If no restrictions were selected, still extract ingredients normally.
- Suggest 2-3 realistic, healthier packaged-food alternatives for "safer_alternatives".
- If the image is unreadable, return empty lists rather than guessing.
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
                        response_mime_type="application/json",
                        temperature=0.2,
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
                        "model returned empty content "
                        f"(finish_reason={finish_reason})"
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
                    f"[OCRService] model "
                    f"'{model_name}' failed:"
                )

                print(
                    traceback.format_exc()
                )

                continue

        # ---------------------------------------------------------
        # 12. ALL MODELS FAILED
        # ---------------------------------------------------------

        print(
            "[OCRService] all Gemini models failed:\n"
            + "\n".join(errors)
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
                "Gemini API call failed, "
                f"showing placeholder data. Last error: {detail}"
            ]
        }
