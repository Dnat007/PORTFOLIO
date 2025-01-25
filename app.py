from flask import Flask, request, render_template, redirect, url_for
from src.pipeline.predictionpipeline import CustomData, PredictPipeline
from src.logger import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from datetime import date
import random

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_brain = models.efficientnet_b0(
    pretrained=False)  # Use same architecture as training
model_state_brain.classifier[1] = nn.Linear(
    model_state_brain.classifier[1].in_features, 4)  # 4 output classes
model_state_brain = model_state_brain.to(device)

# Load the trained model weights
model_path = r"models\\brain_tumor_classifier_.pth"
model_state_brain.load_state_dict(torch.load(model_path, map_location=device))
model_state_brain.eval()  # Set the model to evaluation mode

# Define the image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def home():
    return render_template('indexport.html')


@app.route('/project.html')
def project():
    return render_template('project.html')

# Prediction route


@app.route('/secondpage.html')
def projects():
    return render_template('secondpage.html')


@app.route('/brainprediction', methods=['GET', 'POST'])
def prediction():

    if request.method == 'POST':
        f = request.files['img']
        f.save('img.jpg')

        # Load the image and apply transformations
        img = Image.open('img.jpg').convert('RGB')
        # Add batch dimension and move to device
        img = transform(img).unsqueeze(0).to(device)

        # Make the prediction
        with torch.no_grad():
            output = model_state_brain(img)
        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        indices = predicted_class.item()

        # Map the output to class names
        tumor_types = ['glioma_tumor', 'meningioma_tumor',
                       'no_tumor', 'pituitary_tumor']
        tumor_type = tumor_types[indices]

        confidence = 0.95  # This would be your actual confidence score
        patient_id = random.randint(10000, 99999)
        scan_date = date.today().strftime('%Y-%m-%d')
        return render_template('brainprediction.html', data=tumor_type, name='img.jpg', confidence=confidence,
                               patient_id=patient_id,
                               scan_date=scan_date)


@app.route('/form.html')
def form():
    return redirect(url_for('predict_datapoint'))


@app.route('/predict', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            Airline=request.form.get('Airline'),
            Source=request.form.get('Source'),
            Destination=request.form.get('Destination'),
            Journey_Day=int(request.form.get('Journey_Day')),
            Journey_Month=int(request.form.get('Journey_Month')),
            Journey_Weekday=request.form.get('Journey_Weekday'),
            Departure_Part_of_Day=request.form.get('Departure_Part_of_Day'),
            Arrival_Part_of_Day=request.form.get('Arrival_Part_of_Day'),
            Duration_Hour=int(request.form.get('Duration_Hour')),
            Duration_Min=int(request.form.get('Duration_Min')),
            Total_Stops=request.form.get('Total_Stops')
        )

        final_data = data.get_data_as_dataframe()
        logging.info(f'{final_data}')

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_data)

        # Rounding the prediction value up to 2 points
        result = int(round(pred[0], 2))

        return render_template("result.html", final_result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
