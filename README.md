# 🍃 FoodLens

**Scan a food label, know if it's safe for *you*.**

FoodLens is an ML-powered food ingredient safety scanner. Take a photo of any ingredient label, set up your health profile (diet, allergies, medical conditions), and get an instant safety score, a letter grade, a plain-language explanation of every conflict, and healthier alternatives, in English or Tamil.

---

## ✨ Features

- **Label scanning:** take a photo with your camera or upload one from your gallery
- **AI ingredient extraction:** Google Gemini reads the label, extracts every ingredient, and translates it for display
- **Personalised safety analysis:** a rules engine checks the ingredients against your own profile
- **Safety score and grade:** a 0–100% safety score, a classification (*Safe to Eat / Use with Caution / Not Recommended*) and a Grade A–F
- **Clear explanations:** every conflict is listed, e.g. *"Contains 'milk powder': conflicts with Vegan."*
- **Safer alternatives:** 2–3 healthier packaged-food suggestions
- **Multilingual UI:** English and Tamil toggle in the app (the backend also supports Hindi, Spanish and French)
- **Light / dark theme:** warm, soft "wellness" design in both modes
- **In-app browser detection:** warns users when Instagram, Discord, TikTok and similar browsers block the camera, and offers a copy-link workaround
- **Resilient UX:** loading steps, a 45 s request timeout, and friendly error screens with retry

---

## 🧠 How It Works

```
 ┌──────────┐   photo + profile    ┌────────────────┐
 │ Frontend │ ───────────────────► │  FastAPI       │
 │ (HTML/JS)│                      │  /analyze-food │
 └──────────┘                      └───────┬────────┘
       ▲                                   │
       │                      1. OCRService (Gemini)
       │                         → product name, ingredients,
       │                           translations, alternatives
       │                                   │
       │                      2. RulesEngine
       │                         → matches ingredients against
       │                           rules.json + your profile
       │                                   │
       └──────── JSON result ◄─────────────┘
          score · class · grade · reasons · alternatives
```

1. **Upload:** the frontend sends the image plus the user's profile (age, weight, and every selected restriction) as `multipart/form-data`.
2. **OCR and extraction:** `OCRService` converts the image to JPEG and prompts Gemini to return strict JSON. Ingredients are always extracted in English (used for matching) and also returned translated (used for display).
3. **Rules engine:** `RulesEngine` scores the English ingredient list against the user's active flags.
4. **Response:** the result is returned and rendered as a score ring, ingredient pills, a safety analysis list and alternatives.

### Scoring logic

| Situation | Score | Classification | Grade |
|---|---|---|---|
| Any **allergy / intolerance** match (critical) | 0% | Not Recommended | F |
| 2 or more profile conflicts | 30% | Not Recommended | D |
| Exactly 1 profile conflict | 55% | Use with Caution | C |
| No conflicts | 100% | Safe to Eat | A |
| **No restrictions selected**: general health check | 100% minus penalties | Based on score (≥85 / ≥60 / below) | A / C / D |

The general check subtracts weighted penalties for items like high-fructose corn syrup, trans fats, MSG, sodium nitrate, aspartame, palm oil, sugar and salt.

**False-positive protection:** keyword matching strips known exception phrases first, so *cocoa butter* doesn't trigger a dairy warning, *coconut milk* doesn't trigger milk, and *eggplant* doesn't trigger egg.

---

## 🗂️ Project Structure

```
FoodLence-main/
├── index.html              # Single-page frontend (Tailwind via CDN, vanilla JS)
├── style.css               # Animations, focus states, micro-interactions
└── Backend/
    ├── main.py             # FastAPI app: routes, CORS, request handling
    ├── requirements.txt    # Python dependencies
    ├── services/
    │   └── ocr_service.py  # Gemini client, prompt, JSON parsing, fallback
    ├── rules/
    │   └── engine.py       # Scoring and ingredient-matching logic
    └── datasets/
        ├── rules.json          # Restriction → keyword mappings
        └── ingredients.json    # Canonical ingredient reference data
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, Tailwind CSS (CDN), vanilla JavaScript, Google Fonts (Quicksand, Nunito) |
| Backend | Python, FastAPI, Uvicorn |
| AI / OCR | Google Gemini (`google-genai`) |
| Image handling | Pillow |
| Hosting | Render |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd FoodLence-main/Backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set your API key

```bash
export GEMINI_API_KEY="your-key-here"      # Windows (PowerShell): $env:GEMINI_API_KEY="your-key-here"
```

The app refuses to start without it.

### 3. Run the backend

```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

API is now at `http://localhost:8000`.

### 4. Run the frontend

`main.py` serves the frontend from `Backend/static/index.html`, so copy the frontend files in:

```bash
mkdir -p static
cp ../index.html ../style.css static/
```

To make the page use your **local** backend instead of the hosted one, edit `API_BASE` near the top of the script in `index.html`:

```js
const API_BASE = 'http://localhost:8000';
```

Then open `http://localhost:8000`.

---

## 📡 API Reference

### `POST /analyze-food`

`multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | file | Photo of the ingredient label (required) |
| `lang` | string | `en`, `ta`, `hi`, `es`, `fr` (default `en`) |
| `age`, `weight` | int, float | Optional user metrics |
| Dietary | bool | `vegan`, `vegetarian`, `keto`, `paleo`, `halal`, `kosher`, `gluten_free`, `dairy_free` |
| Allergies | bool | `peanuts`, `tree_nuts`, `milk_dairy`, `eggs`, `soy`, `fish`, `shellfish`, `sesame`, `wheat`, `mustard` |
| Health | bool | `diabetic`, `high_blood_pressure`, `lactose_intolerant`, `celiac_disease`, `high_cholesterol`, `kidney_care` |

**Response**

```json
{
  "product_name": "Chocolate Biscuits",
  "detected_ingredients": ["Wheat Flour", "Sugar", "Milk Powder", "Salt"],
  "safer_alternatives": ["Oat & Date Cookies", "Whole Wheat Digestives"],
  "safety_percentage": 0.0,
  "classification": "Not Recommended",
  "nutri_grade": "Grade F",
  "recommendations": [
    "Contains 'Milk Powder': conflicts with Milk Dairy."
  ]
}
```

Errors return `{"error": "..."}` with status `400` (empty upload) or `500`.

### `GET /`

Serves the frontend.

---

## 🛠️ Customising

- **Add or edit restrictions:** edit `Backend/datasets/rules.json`. Anything under `allergies_and_intolerances` is treated as *critical* (forces Grade F).
- **Fix false positives:** add phrases to `EXCEPTION_PHRASES` in `rules/engine.py`.
- **Tune the general health penalties:** edit `GENERAL_RISK_WEIGHTS` in `rules/engine.py`.
- **Change the Gemini model:** edit `MODEL_CANDIDATES` in `services/ocr_service.py`. Add more entries to get automatic fallback between models.
- **Add a language:** add it to `LANGUAGES` in `ocr_service.py` and to the `I18N` object in `index.html`.

---

## ⚠️ Known Limitations

- If the Gemini call fails, `OCRService` returns **placeholder demo ingredients** (water, sugar, wheat flour, milk powder, salt) with the error shown in the alternatives list, so results in that state are not real.
- Some profile options in the UI (`crustaceans`, `molluscs`, `gluten_cereals`, `celery`, `lupin`, `sulphites`, and the fitness goals) are not yet accepted by the `/analyze-food` endpoint, so they currently have no effect on the score.
- Matching is keyword-based, so unusual ingredient names or very short labels may be missed.
- The UI toggle offers English and Tamil only, though the backend supports five languages.
- CORS is open (`*`) and there is no rate limiting. Tighten both before a public launch.

---

## 🩺 Disclaimer

FoodLens is an educational project and **not a substitute for professional medical or dietary advice**. AI-based label reading can make mistakes. If you have a severe allergy, always read the physical label and consult a healthcare professional.
---