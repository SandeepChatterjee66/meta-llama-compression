from flask import Flask, session, render_template, request, redirect, url_for, jsonify
import random, os, secrets
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

secret_key = secrets.token_urlsafe(16)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

# Set upload folder path and allowed extensions for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

                # Store the selected model and filename in the session
                session['selected_model'] = selected_model
                session['filename'] = filename

                return redirect(url_for('progress'))
            except Exception as e:
                return f"Error uploading file: {str(e)}", 500
        else:
            return "Invalid file type", 400
    
    # Render the index page with model carousel
    return render_template('index.html')





# Load TinyBERT model and tokenizer
tinybert_model = AutoModelForSequenceClassification.from_pretrained('google/tinybert-uncased-L-4-H-128')
tinybert_tokenizer = AutoTokenizer.from_pretrained('google/tinybert-uncased-L-4-H-128')

# Route for the progress page (asynchronous updates can be added here)
@app.route('/progress')
def progress():
    # Retrieve the selected model and filename from the session
    selected_model = session.get('selected_model')
    filename = session.get('filename')
    
    # Load the CSV file
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(csv_file_path)
    
    # Initialize progress and row counter
    progress_value = 0
    row_counter = 0
    
    # Process the CSV file row by row
    for index, row in df.iterrows():
        # Pass the row to TinyBERT model
        inputs = tinybert_tokenizer(row['text'], return_tensors='pt')
        outputs = tinybert_model(**inputs)
        
        # Update progress
        row_counter += 1
        progress_value = (row_counter / len(df)) * 100
        
        # Store the progress value in the session
        session['progress_value'] = progress_value
        
        # Simulate processing time (optional)
        import time
        time.sleep(0.1)
    
    return render_template('progress.html', selected_model=selected_model, filename=filename)

# API route to update progress
@app.route('/progress/update', methods=['GET'])
def update_progress():
    progress_value = session.get('progress_value', 0)
    return jsonify(progress=progress_value)




# # Route for the progress page (asynchronous updates can be added here)
# @app.route('/progress')
# def progress():
#     # Retrieve the selected model and filename from the session
#     selected_model = session.get('selected_model')
#     filename = session.get('filename')

#     return render_template('progress.html', selected_model=selected_model, filename=filename)

# progress_value = 0

# def update_progress_value():
#     global progress_value
#     progress_value += 10  # Increase progress by 10% each second
#     if progress_value > 100:
#         progress_value = 100

# # API route to update progress
# @app.route('/progress/update', methods=['GET'])
# def update_progress():
#     update_progress_value()
#     return jsonify(progress=progress_value)

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

# Route to handle chat messages (simulated LLM response for now)
@app.route('/chat/send', methods=['POST'])
def chat_send():
    user_message = request.json['message']
    # Simulated LLM response (this will eventually be replaced with actual model inference)
    llm_response = "This is a response to: " + user_message
    return jsonify(response=llm_response)

if __name__ == '__main__':
    app.run(debug=True)