import os

# Define the folder structure
folders = [
    "./static/css",
    "./static/images",
    "./static/js",
    "./templates"
]

# Define the files to create
files = {
    "./static/css/styles.css": "",
    "./static/images/meta_logo.png": "",  # Placeholder for image
    "./static/js/script.js": "",
    "./templates/index.html": "",
    "./templates/progress.html": "",
    "./templates/heatmap.html": "",
    "./templates/compress.html": "",
    "./templates/chat.html": ""
}

# Create the folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created directory: {folder}")

# Create the files
for filepath, content in files.items():
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created file: {filepath}")
