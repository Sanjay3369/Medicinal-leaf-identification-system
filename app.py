import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template_string
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model safely
model_path = r"E:\capstone data\fixed_model.keras"

try:
    model = tf.keras.models.load_model(model_path, compile=False)
except TypeError:
    # Attempting alternate deserialization
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'InputLayer': tf.keras.layers.InputLayer})

# Load suggestions from Excel
excel_path = r"E:\Indian_Medicinal_Leaves_Dataset\plant_classes.xlsx"
df_suggestions = pd.read_excel(excel_path)

# Set up validation directory to extract class labels
val_dir = r"E:\Indian_Medicinal_Leaves_Dataset\Val"
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

if os.path.exists(val_dir):
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Extract class indices and map them to labels
    class_indices = val_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
else:
    print("Validation directory not found! Using a default mapping.")
    class_labels = {}  # Fallback if validation data isn't available


def predict_and_suggest(image_path):
    """Predict the class of the given image and fetch suggestions."""
    img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels.get(predicted_class_index, "Unknown")

    # Fetch all suggestions for the predicted class
    suggestion_row = df_suggestions[df_suggestions.iloc[:, 0] == predicted_class_label]

    # Extract column-wise data as a dictionary
    suggestion_dict = {}
    if not suggestion_row.empty:
        for col in df_suggestions.columns[1:]:
            values = suggestion_row[col].dropna().values
            suggestion_dict[col] = values if len(values) > 0 else ["No suggestions available."]
    else:
        suggestion_dict = {"Note": ["No suggestions available for this class."]}

    confidence_score = np.max(predictions) * 100
    return predicted_class_label, suggestion_dict, confidence_score


def render_suggestions(suggestion_data):
    """Render suggestion data as HTML content."""
    suggestion_html = ""
    if isinstance(suggestion_data, dict):
        for column_name, values in suggestion_data.items():
            suggestion_html += f"<strong>{column_name}:</strong><ul>"
            for value in values:
                if "http" in value:
                    links = re.split(r'\s+', value.strip())
                    for link in links:
                        if link.startswith("http"):
                            suggestion_html += f'<li><a href="{link}" target="_blank">{link}</a></li>'
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', value.strip())
                    for sentence in sentences:
                        suggestion_html += f"<li>{sentence}</li>"
            suggestion_html += "</ul>"
    return suggestion_html


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and display predictions."""
    result, suggestion_data, confidence_score = "", "", 0
    image_display_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"

        file = request.files['file']
        if file.filename == '':
            return "No file selected!"

        temp_file_path = "static/uploaded_image.jpg"
        file.save(temp_file_path)

        # Perform prediction
        result, suggestion_data, confidence_score = predict_and_suggest(temp_file_path)
        image_display_path = temp_file_path

    suggestion_html = render_suggestions(suggestion_data)

    html_content = f'''
    <!doctype html>
    <html>
    <head>
        <title>Medical Plant/Leaf Classification</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                padding: 20px;
                max-width: 1200px;
                margin: auto;
                background-color: #ffffff;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                display: grid;
                grid-template-columns: 1.5fr 2fr;
                gap: 20px;
            }}
            .left-panel {{
                background-color: #e0f7fa;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            }}
            .right-panel {{
                background-color: #e8f5e9;
                border-radius: 10px;
                padding: 20px;
                height: 100%;
                max-height: 600px;
                overflow-y: auto;
            }}
            h1 {{ color: #00796b; }}
            img {{
                max-width: 100%;
                border-radius: 10px;
                margin-top: 10px;
                max-height: 250px;
            }}
            ul {{ list-style-type: square; padding-left: 20px; }}
            a {{ color: #00796b; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .result-card {{
                padding: 10px;
                background-color: #d7f3e5;
                border-radius: 10px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="left-panel">
                <h1>Medical Plant/Leaf Classification</h1>
                <form method="post" enctype="multipart/form-data">
                    <input type="file" name="file" style="margin-top: 10px;"><br>
                    <input type="submit" value="Upload" style="margin-top: 10px;">
                </form>

                {'<img src="' + image_display_path + '" alt="Uploaded Image">' if image_display_path else ''}

                <div class="result-card">
                    <h2>Prediction Result: {result}</h2>
                    <p><strong>Confidence Score:</strong> {confidence_score:.2f}%</p>
                </div>
            </div>

            <div class="right-panel">
                <h2>Suggestions</h2>
                {suggestion_html}
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html_content)


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
