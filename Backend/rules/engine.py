import os
import json


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Plain substring matching is prone to false positives when a short keyword
# is embedded in an unrelated compound term — e.g. the dairy keyword
# "butter" also appears inside "cocoa butter" and "shea butter", and "milk"
# appears inside "coconut milk". Whenever one of these keywords is checked,
# any of its listed exception phrases are stripped out of the ingredient
# text first, so the keyword can no longer match inside them.
EXCEPTION_PHRASES = {
    "butter": ["cocoa butter", "cacao butter", "shea butter", "mango butter",
               "illipe butter", "cashew butter", "almond butter",
               "sunflower butter", "coconut butter", "peanut butter"],
    "milk": ["coconut milk", "soy milk", "soya milk", "oat milk", "almond milk",
             "rice milk", "cashew milk", "hemp milk", "pea milk", "flax milk"],
    "cream": ["coconut cream", "cream of tartar", "tartar cream"],
    "egg": ["eggplant"],
    "eggs": ["eggplant"],
}


class RulesEngine:
    """Loads rules.json / ingredients.json and scores a list of ingredients
    against a user's selected health/dietary profile."""

    # General "no restrictions selected" health-risk keyword weights
    GENERAL_RISK_WEIGHTS = {
        "high fructose corn syrup": 25, "hfcs": 25, "hydrogenated oil": 25,
        "trans fat": 25, "partially hydrogenated oil": 25,
        "monosodium glutamate": 20, "msg": 20, "sodium nitrate": 20,
        "aspartame": 20, "palm oil": 15, "lard": 20,
        "sugar": 15, "sucrose": 15, "glucose": 15, "fructose": 15,
        "salt": 12, "sodium": 12, "gelatin": 10,
    }

    def __init__(self, base_dir: str = None):
        base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rules_config = _load_json(os.path.join(base_dir, "datasets", "rules.json"))
        self.ingredients_db = _load_json(os.path.join(base_dir, "datasets", "ingredients.json"))

        # Anything under allergies_and_intolerances is treated as a "critical"
        # rule: a single match forces the product to Grade F / 0%.
        self.critical_rule_keys = set(self.rules_config.get("allergies_and_intolerances", {}).keys())

    @staticmethod
    def _keyword_matches(kw_l: str, norm: str) -> bool:
        """Substring match with exception phrases stripped out first, so a
        short keyword can't false-positive inside an unrelated compound
        ingredient name (e.g. 'butter' inside 'cocoa butter')."""
        exceptions = EXCEPTION_PHRASES.get(kw_l)
        if not exceptions:
            return kw_l in norm
        cleaned = norm
        for exc in exceptions:
            cleaned = cleaned.replace(exc, " ")
        return kw_l in cleaned

    def analyze(self, ingredients_list: list, user_profile: dict) -> dict:
        if not ingredients_list:
            return {
                "safety_percentage": 0.0,
                "classification": "Analysis Failed",
                "nutri_grade": "Grade F",
                "recommendations": ["No ingredients could be detected from the image."],
            }

        norm_ings = [str(i).strip().lower() for i in ingredients_list]
        active_flags = {k for k, v in user_profile.items() if v is True and k not in ("age", "weight")}

        # ---- 1. No restrictions selected -> general health assessment ----
        if not active_flags:
            return self._general_assessment(ingredients_list, norm_ings)

        # ---- 2. Profile-based restriction checks ----
        reasons = []
        has_critical = False

        for category, rules in self.rules_config.items():
            for rule_key, keywords in rules.items():
                if rule_key not in active_flags:
                    continue
                for kw in keywords:
                    kw_l = kw.lower()
                    for raw, norm in zip(ingredients_list, norm_ings):
                        if self._keyword_matches(kw_l, norm):
                            msg = f"Contains '{raw}' — conflicts with {rule_key.replace('_', ' ').title()}."
                            if msg not in reasons:
                                reasons.append(msg)
                            if rule_key in self.critical_rule_keys:
                                has_critical = True
                            break

        if has_critical:
            score, cls, grade = 0.0, "Not Recommended", "Grade F"
        elif len(reasons) >= 2:
            score, cls, grade = 30.0, "Not Recommended", "Grade D"
        elif len(reasons) == 1:
            score, cls, grade = 55.0, "Use with Caution", "Grade C"
        else:
            score, cls, grade = 100.0, "Safe to Eat", "Grade A"

        return {
            "safety_percentage": score,
            "classification": cls,
            "nutri_grade": grade,
            "recommendations": reasons if reasons else ["No conflicts found with your selected health profile."],
        }

    def _general_assessment(self, ingredients_list, norm_ings):
        warnings = []
        score = 100.0
        for raw_ing, norm_ing in zip(ingredients_list, norm_ings):
            for bad_key, penalty in self.GENERAL_RISK_WEIGHTS.items():
                if bad_key in norm_ing:
                    score -= penalty
                    msg = f"Contains '{raw_ing}' — high in sugar/sodium/additives."
                    if msg not in warnings:
                        warnings.append(msg)

        score = max(0.0, round(score, 1))

        if score >= 85.0:
            classification, grade = "Safe to Eat (Clean Ingredients)", "Grade A"
        elif score >= 60.0:
            classification, grade = "Use with Caution (Processed)", "Grade C"
        else:
            classification, grade = "Not Recommended (Unhealthy)", "Grade D"

        return {
            "safety_percentage": score,
            "classification": classification,
            "nutri_grade": grade,
            "recommendations": warnings if warnings else ["Ingredients appear standard — no major health flags."],
        }