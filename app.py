import cv2
import numpy as np
from flask import Flask, request, jsonify
from views import views
import os
from fatsecret import Fatsecret
import spacy
import requests
from spellchecker import SpellChecker
import re
import shelf_life as sl
import csv
from dotenv import load_dotenv

load_dotenv(dotenv_path="api.env")

nlp = spacy.load("en_core_web_md")

fs = Fatsecret(os.getenv("fs1"), os.getenv("fs2"))

SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", os.getenv("Spoonacular"))
SPOONACULAR_BASE_URL = "https://api.spoonacular.com/recipes/findByIngredients"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/views")

def ensure_file_size_under_limit(file_path, max_size_kb=1024):
    try:
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb <= max_size_kb:
            return file_path

        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Could not read image for resizing")

        quality = 90
        temp_path = os.path.splitext(file_path)[0] + "_compressed.jpg"

        while quality > 10:
            cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if os.path.getsize(temp_path) / 1024 <= max_size_kb:
                return temp_path
            quality -= 10

        height, width = img.shape[:2]
        scale = 0.9
        while os.path.getsize(temp_path) / 1024 > max_size_kb and scale > 0.1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(temp_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            scale -= 0.1

        return temp_path

    except Exception as e:
        print(f"Error in file compression: {e}")
        return file_path

def ocr_space_api(file_path):
    try:
        with open(file_path, 'rb') as file:
            payload = {
                'apikey': os.getenv("ocr_space"),
                'isOverlayRequired': False,
                'language': 'eng',
                'OCREngine': 2,
            }
            response = requests.post('https://api.ocr.space/parse/image', files={'file': file}, data=payload)
            response.raise_for_status()
            result = response.json()
            print("result",result)
            if result.get("OCRExitCode") == 1:
                parsed_text = result["ParsedResults"][0]["ParsedText"]
                print("parsed_text", parsed_text.split('\n'))
                return parsed_text.split('\n')
            elif 'E101: Timed out waiting for results' in result.get('ErrorMessage', []):
                return backup_ocr_space_api(file_path)
            else:
                return []

    except Exception as e:
        print(f"Error querying OCR.space API: {e}")
        return []

def backup_ocr_space_api(file_path):
    try:
        with open(file_path, 'rb') as file:
            payload = {
                'apikey': os.getenv("ocr_space"),
                'isOverlayRequired': False,
                'language': 'eng',
                'OCREngine': 1,
            }
            response = requests.post('https://api.ocr.space/parse/image', files={'file': file}, data=payload)
            response.raise_for_status()
            result = response.json()

            if result.get("OCRExitCode") == 1:
                parsed_text = result["ParsedResults"][0]["ParsedText"]
                return parsed_text.split('\n')
            else:
                return []

    except Exception as e:
        print(f"Error querying OCR.space API: {e}")
        return []

def load_food_descriptions(csv_path='data/cleaned_ingredients.csv'):
    food_words = set()
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                description = row.get('Descrip')
                if description:
                    words = description.lower().split()
                    food_words.update(words)
    except Exception as e:
        print(f"Error loading food dictionary: {e}")
    return food_words

def postprocess_text(detected_text):
    if not detected_text:
        return []

    # Extract words from end of each line
    filtered_text = []
    for line in detected_text:
        match = re.search(r'([A-Za-z]+)$', line.strip())
        if match:
            filtered_text.append(match.group(1).lower())

    # Load custom food words from CSV
    custom_words = load_food_descriptions()

    spell = SpellChecker()
    spell.word_frequency.load_words(custom_words)
    # Correct spelling using custom vocabulary
    cleaned_text = [
        spell.correction(word) if word not in spell else word
        for word in filtered_text
    ]

    return cleaned_text

def search_recipes_by_ingredients(ingredients):
    try:
        params = {
            "ingredients": ",".join(ingredients),
            "number": 5,
            "apiKey": SPOONACULAR_API_KEY
        }
        response = requests.get(SPOONACULAR_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching recipes: {e}")
        return []

def foodSearch(food):
    try:
        foods = fs.foods_search(food)
        if foods:
            return foods[0]["food_name"]
    except Exception:
        pass
    return None

def is_semantically_similar(word1, word2, threshold=0.8):
    try:
        token1 = nlp(word1.lower())
        token2 = nlp(word2.lower())

        if not token1.vector_norm or not token2.vector_norm:
            return False

        return token1.similarity(token2) >= threshold
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return False

def process_receipt(path):
    detected_text = ocr_space_api(path)
    processed_text = postprocess_text(detected_text)
    return processed_text

@app.route('/upload', methods=['PUT'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        original_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(original_path)

        file_path = ensure_file_size_under_limit(original_path)

        if file_path != original_path and os.path.exists(original_path):
            os.remove(original_path)

        ocr_result = process_receipt(file_path)
        detected_foods = []
        shelf_life = []

        for text in ocr_result:
            print(text)
            food_name = foodSearch(text)
            if food_name and is_semantically_similar(text, food_name):
                detected_foods.append(food_name)
        print(detected_foods)
        for i in detected_foods:
            shelf_life.append(sl.print_shelf_life(i))

        detected_foods = list(set(detected_foods))

        recipes = search_recipes_by_ingredients(detected_foods)
        formatted_recipes = [{"title": recipe.get("title"), "image": recipe.get("image")} for recipe in recipes]
        return jsonify({
            "message": "File uploaded and processed successfully!",
            "detected_foods": detected_foods,
            "recipes": formatted_recipes,
            "shelf_life": shelf_life
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)