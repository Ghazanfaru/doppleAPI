import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import cv2

# Initialize the Flask app
app = Flask(__name__)

# Define your Excel file and sheet name
xlsx_path = 'Pahtbook.xlsx'
sheet_name = 'Sheet1'

# Load data from the Excel file into a DataFrame
data_frame = pd.read_excel(xlsx_path, sheet_name=sheet_name)

# Load database images when the app starts
def load_database_images():
    database_urls = data_frame['IMAGE']
    database_images = []
    for image_url in database_urls:
        image = load_and_preprocess_image_from_url(image_url)
        if image is not None and image.shape == (224, 224, 3):
            database_images.append(image)
    return database_images

def load_and_preprocess_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(BytesIO(response.content))
        image = image.resize((224, 224))  # Resize the image to your desired dimensions
        image = img_to_array(image) / 255.0  # Normalize pixel values
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from URL {image_url}: {e}")
        return None

# Calculate similarity between two images
def calculate_similarity(input_image, database_image):
    return np.mean((input_image - database_image) ** 2)

# Define a route for your endpoint
@app.route('/get_dopple', methods=['POST'])
def get_dopple():
    input_image_url = request.form.get('image_url')
    input_image = load_and_preprocess_image_from_url(input_image_url)

    if input_image is None:
        return jsonify({"error": "Error loading input image"}), 400

    if input_image.shape != (224, 224, 3):
        return jsonify({"error": "Input image dimensions not valid"}), 400

    # Load database images when the app starts
    database_images = load_database_images()

    # Calculate similarity scores with all database images
    similarity_scores = []
    for database_image in database_images:
        similarity_score = calculate_similarity(input_image, database_image)
        similarity_scores.append(similarity_score)

    # Get the indices of the top 10 most similar images
    top_indices = np.argsort(similarity_scores)[:10]

    # Create a list of similar image names and corresponding data
    similar_images_info = []
    for index in top_indices:
        images = data_frame.loc[index, 'IMAGE']
        name = data_frame.loc[index, 'NAME']
        department = data_frame.loc[index, 'DEPARTMENT']
        graduation_year = data_frame.loc[index, 'GRADUATION-YEAR']
        location = data_frame.loc[index, 'LOCATION']
        similar_images_info.append({
            'image_url': images,
            "name": name,
            "department": department,
            "graduation_year": graduation_year,
            "location": location
        })

    return jsonify(similar_images_info)

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=False)
