from flask import Flask, render_template, request, url_for # import the Flask, render_template, request, and url_for classes from the flask package
import tensorflow as tf# import the tensorflow package with an alias of tf
import numpy as np# import the numpy package with an alias of np
import os # import the os package

app = Flask(__name__, static_url_path = "/assets", static_folder = "assets")

model_directory = "models" # set the model directory

classes = [ "Adelie", "Gentoo", "Chinstrap" ]

@app.route("/") # set the index.html route to /
def index():
    model_files = []

    for current_file in os.listdir(model_directory):
        model_files.append(os.path.join(model_directory, current_file))

    model_files.sort(key = os.path.getmtime, reverse = True)

    return render_template("index.html", model_options = "".join(f"<option value=\"{os.path.basename(current_model_file)}\">{os.path.basename(current_model_file)}</option>" for current_model_file in model_files)) # render the index.html webpage

@app.route("/predict", methods = ["POST"]) # set the predict.html route to /predict that accepts a POST request
def predict():
    # Fill in the missing features from the form fields.
    features_form = [
        float(request.form["culmen-length"]),
        float(request.form["culmen-depth"]),
        float(request.form["flipper-length"]),
        float(request.form["body-mass"])

    ]

    model = tf.keras.models.load_model(os.path.join(model_directory, request.form['model'])) # load the selected model

    class_probabilities = model.predict(np.array([features_form])) # predict the model with a numpy-based array of the features

    best_class_index = np.argmax(class_probabilities, axis = 1)[0]

    class_predicted = classes[best_class_index]

    return render_template("predict.html", class_predicted = class_predicted, class_image_url = url_for("static", filename = f"{class_predicted.lower()}.jpg")) # render the predict.html webpage with the predicted class and the lowercased file name of the predicted class

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True) # run the Flask app server on IP address 0.0.0.0, add debug = True to enable debugging
