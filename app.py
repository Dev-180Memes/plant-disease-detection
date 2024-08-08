from flask import Flask, request, jsonify
from fastai.vision.all import load_learner, PILImage
import io

app = Flask(__name__)

# Load Fastai models
cucumber_model = load_learner('models/cucumber_classifier_model.pkl')
pumpkin_model = load_learner('models/pumpkin_classifier_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = PILImage.create(io.BytesIO(file.read()))

    model_choice = request.form.get('model')
    if model_choice == 'cucumber':
        prediction, _, probs = cucumber_model.predict(image)
    elif model_choice == 'pumpkin':
        prediction, _, probs = pumpkin_model.predict(image)
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
