from flask import Flask, session, render_template, request, redirect, url_for, jsonify
import random, os, secrets, threading, requests, json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

secret_key = secrets.token_urlsafe(16)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

# Load lightweight tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Set upload folder path and allowed extensions for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/intermediate')
def intermediate():
    random_values = [random.random() < 0.1 for _ in range(26)]  # Generate 26 random values
    return render_template('intermediate.html', random_values=random_values)

# Route for the main page (model carousel and CSV upload)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        
        # If user does not select file, browser may also submit an empty part
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            try:
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"Saving file to: {file_path}")
                file.save(file_path)

                # Get the selected model from the request form data
                selected_model = request.form.get('selected_model')

                # Map the selected model to a base model
                base_models = {
                    'Model 1': 'distilbert-base-uncased',
                    'Model 2': 'bert-base-uncased',
                    'Model 3': 'roberta-base',
                    # Add more models as needed
                }
                session['base_model'] = base_models[selected_model]
                session['selected_model'] = selected_model
                session['filename'] = filename

                return redirect(url_for('process_csv'))
            except Exception as e:
                return f"Error uploading file: {str(e)}", 500
        else:
            return "Invalid file type", 400
    
    # Render the index page with model carousel
    return render_template('index.html')

# Function to process CSV file in chunks
def process_csv_in_chunks(filename):
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(csv_file_path)

    # Load the selected base model
    base_model_name = session.get('base_model')
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    session['progress_value'] = 0
    session['row_counter'] = 0
    
    chunk_size = 10

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]

        for index, row in chunk.iterrows():
            text_column = str(row.iloc[0])

            # Preprocess the text
            inputs = tokenizer(text_column, return_tensors='pt')

            # Perform the forward pass
            outputs = model(**inputs)

            # Process the outputs as needed

            session['row_counter'] += 1
            session['progress_value'] = (session['row_counter'] / len(df)) * 100
            
            session.modified = True  # Ensure session is updated
            
            import time
            time.sleep(0.1)

# Route for processing the CSV file
@app.route('/process-csv', methods=['POST', 'GET'])
def process_csv():
    filename = session.get('filename')
    
    thread = threading.Thread(target=process_csv_in_chunks, args=(filename,))
    thread.start()
    
    return redirect(url_for('progress_page'))

# Route for progress updates
@app.route('/progress')
def progress():
    progress_value = session.get('progress_value', 0)
    return jsonify({'progress': progress_value})

# Route for progress page
@app.route('/progress-page', methods=['GET'])
def progress_page():
    selected_model = session.get('selected_model')
    filename = session.get('filename')
    
    return render_template('progress.html', selected_model=selected_model, filename=filename)

# Route for heatmap (with random values)
@app.route('/heatmap')
def heatmap():
    # Generate a 22x22 grid of random values between 0 and 100
    heatmap_values = [[random.randint(0, 100) for _ in range(22)] for _ in range(22)]
    return render_template('heatmap.html', heatmap_values=heatmap_values)

# Route for the model compression page (placeholder)
@app.route('/compress')
def compress():
    return render_template('compress.html')

# Route for the chat page
@app.route('/chat')
def chat():
    return render_template('chat.html')

import math

@app.route('/pricing')
def pricing():
    # Simulated computation logic
    compute_power = random.uniform(15, 20)  # 1-5 hours
    dataset_size = random.randint(500000, 2000000)  # 0.5-2 million tokens
    compressed_model_size = random.randint(400000000, 1000000000)  # 0.4-1 billion parameters
    finetuning_rate = 0.0001  # ₹0.0001 per parameter
    # finetuning_cost = abs(dataset_size - compressed_model_size) * finetuning_rate
    total_cost = random.uniform(200, 300)  # ₹200-300
    finetuning_cost = total_cost - 50;

    return render_template('pricing.html',
                           compute_power=round(compute_power, 2),
                           dataset_size=dataset_size,
                           compressed_model_size=compressed_model_size,
                           finetuning_cost=round(finetuning_cost, 2),
                           total_cost=round(total_cost, 2))

# Load the original model
original_model_name = 'llama3.2:1b'
original_model = AutoModelForSequenceClassification.from_pretrained(original_model_name)
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)

# Load the compressed model
compressed_model_path = 'path/to/compressed/model'  # Update this path
compressed_model = AutoModelForSequenceClassification.from_pretrained(compressed_model_path)
compressed_tokenizer = AutoTokenizer.from_pretrained(original_model_name)  # Use the same tokenizer as the original model

# Move the compressed model to the correct device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compressed_model.to(device)

@app.route('/chat/send/original', methods=['POST'])
def handle_chat_request_original():
    data = request.get_json()
    user_message = data['message']+', answer in one sentence or if possible in one word'

    # Preprocess the text
    inputs = original_tokenizer(user_message, return_tensors='pt')

    # Perform inference with the original model
    outputs = original_model(**inputs)

    # Process the outputs as needed
    response = original_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response, 'context': ""})

@app.route('/chat/send/compressed', methods=['POST'])
def handle_chat_request_compressed():
    data = request.get_json()
    user_message = data['message']+', answer in one sentence or if possible in one word'

    # Preprocess the text
    inputs = compressed_tokenizer(user_message, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device

    # Perform inference with the compressed model
    outputs = compressed_model(**inputs)

    # Process the outputs as needed
    response = compressed_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response, 'context': ""})

@app.route('/impact')
def impact():
    return render_template('impact.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)