import os
import argostranslate.package
import argostranslate.translate

# Step 1: Set the local model file path (update with your exact path)
local_model_path = r"D:\Langchain\Project\Final\argos_models\translate-en_hi-1_1.argosmodel"

# Step 2: Function to install the model if not installed
def install_model_if_needed(model_path):
    installed_languages = argostranslate.translate.get_installed_languages()
    installed_codes = [lang.code for lang in installed_languages]

    # Check if English->Hindi translation already installed
    if "en" in installed_codes and "hi" in installed_codes:
        en_lang = next(l for l in installed_languages if l.code == "en")
        hi_lang = next(l for l in installed_languages if l.code == "hi")
        for translation in en_lang.translations:
            if translation.to_lang.code == "hi":
                print("English to Hindi translation model already installed.")
                return True

    # Install from local package file
    print("Installing English to Hindi translation model from local file...")
    argostranslate.package.install_from_path(model_path)
    print("Installation complete.")
    return True

# Step 3: Offline translate function
def offline_translate_to_hindi(text: str) -> str:
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in installed_languages if l.code == "en"), None)
    to_lang = next((l for l in installed_languages if l.code == "hi"), None)

    if from_lang is None or to_lang is None:
        return "Hindi translation model not installed."

    translation = from_lang.get_translation(to_lang)
    translated_text = translation.translate(text)
    return translated_text

# === Main execution ===

if os.path.exists(local_model_path):
    install_model_if_needed(local_model_path)
else:
    print("Model file not found! Please check the path.")

# Test translation
sample_text = "Hello, how are you?"
translated = offline_translate_to_hindi(sample_text)
print(f"Original: {sample_text}")
print(f"Translated: {translated}")
