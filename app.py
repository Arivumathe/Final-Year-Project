from flask import Flask, render_template, request, redirect
from main import predict
import os

# UPLOAD_FOLDER = 'static'  # Create a folder named 'uploads' in the same directory as app.py

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # Function to process the image and return a text response
def process_image(image):
    # Save the uploaded image to the upload folder
    filename = image.filename
    # filepath = os.path.join("uploads", filename)
    image.save(os.path.join("static",filename))
    print(filename)
    # Return the image filename
    return predict(filename),filename

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image from the request
        image = request.files['image']

        # Process the image and get the filename
        output,image_filename = process_image(image)

        # Render the template with the text response and image filename
        return render_template('result.html', text_response="Image uploaded successfully.", image_filename=image_filename,output=output)

    # Render the upload form template for GET requests
    return render_template('index.html')

# @app.route('/reset', methods=['POST'])
# def reset_image():
#     return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
