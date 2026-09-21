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


# Use models supported by the new google-genai SDK.
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def _strip_code_fence(text: str) -> str:
    """
    Some models may return JSON inside ```json ... ``` even when
    instructed not to. Remove the code fence before parsing.
    """

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

        # Get the API key from the parameter or environment variable.
        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if api_key:
            api_key = api_key.strip()

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Set it as an environment variable "
                "(Render: Dashboard -> your service -> Environment) "
                "instead of hardcoding it in source."
            )

       
        self.client = genai.Client(api_key=api_key)

        self.models = MODEL_CANDIDATES


        if len(api_key) > 8:
            masked = (
                f"{api_key[:4]}..."
                f"{api_key[-4:]}"
                f" (length {len(api_key)})"
            )
        else:
            masked = "(too short to mask safely)"

        print(
            f"[OCRService] Loaded GEMINI_API_KEY: {masked}"
        )

    def analyze_image(
        self,
        image_bytes: bytes,
        user_profile: dict,
        lang: str = "en"
    ) -> dict:

   

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

       
        buf = io.BytesIO()

        image.save(
            buf,
            format="JPEG"
        )

        jpeg_bytes = buf.getvalue()

        # ---------------------------------------------------------
        # 2. Determine requested language
        # ---------------------------------------------------------

        target_lang = LANGUAGES.get(
            lang,
            "English"
        )

        # ---------------------------------------------------------
        # 3. Get selected user restrictions
        # ---------------------------------------------------------

        active = [
            k.replace("_", " ").title()
            for k, v in user_profile.items()
            if v is True
            and k not in ("age", "weight")
        ]

        restrictions_str = (
            ", ".join(active)
            if active
            else "None (general health assessment only)"
        )

        # ---------------------------------------------------------
        # 4. Gemini prompt
        # ---------------------------------------------------------

        prompt = f"""
You are an OCR and ingredient-extraction assistant
for a food label scanner.

Look at the attached food label photo and read every
ingredient listed.

USER'S SELECTED HEALTH/DIETARY RESTRICTIONS:
{restrictions_str}

Return ONLY valid JSON.

Do not return markdown.
Do not return ```json.
Do not return explanations.
Do not return any text outside the JSON.

Use this exact schema:

{{
  "product_name": "product name in English",
  "product_name_translated": "product name translated into {target_lang}",
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
    "alternative product 2 in {target_lang}"
  ]
}}

Rules:

- "detected_ingredients" MUST stay in English.
- The English ingredient list is used by the automated
  Food Lens rule-matching system.
- "detected_ingredients_translated" must contain the
  same ingredients translated into {target_lang}.
- If no restrictions were selected, still extract
  the ingredients normally.
- Suggest 2-3 realistic healthier packaged-food
  alternatives for "safer_alternatives".
- If the image is unreadable, return empty ingredient
  lists rather than guessing.
- Do not invent ingredients that cannot be read.
"""

        image_part = types.Part.from_bytes(
            data=jpeg_bytes,
            mime_type="image/jpeg"
        )

       

        errors = []

        for model_name in self.models:

            try:

                print(
                    f"[OCRService] Trying model: {model_name}"
                )

                # NEW google-genai SDK call
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        image_part,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )

              
                raw = (response.text or "").strip()

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
                        "Model returned empty content "
                        f"(finish_reason={finish_reason})"
                    )

                print(
                    f"[OCRService] Model {model_name} "
                    "returned a response."
                )



                cleaned = _strip_code_fence(raw)

                data = json.loads(cleaned)

               
                return self._normalize(data)

            except Exception as exc:

                error_message = (
                    f"{model_name}: {exc}"
                )

                errors.append(error_message)

                print(
                    f"[OCRService] Model "
                    f"'{model_name}' failed:"
                )

                print(
                    traceback.format_exc()
                )

                # Try the next model.
                continue


        print(
            "[OCRService] All Gemini models failed:"
        )

        for error in errors:
            print(error)

        return self._fallback(errors)

    @staticmethod
    def _normalize(data: dict) -> dict:

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

   

    @staticmethod
    def _fallback(errors: list) -> dict:

        detail = (
            errors[-1]
            if errors
            else "Unknown error"
        )

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
                "Salt",
            ],

            "detected_ingredients_translated": [
                "Water",
                "Sugar",
                "Wheat Flour",
                "Milk Powder",
                "Salt",
            ],

            "safer_alternatives": [
                "Gemini API call failed, showing placeholder "
                f"data. Last error: {detail}"
            ],
        }
