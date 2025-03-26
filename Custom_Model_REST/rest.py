from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from customModel import CustomNetwork, CustomDataset, load_images, evaluate_model, image_folder, transform_test


app = Flask(__name__)

model = CustomNetwork()
model.load_state_dict(torch.load('modelWeights/model_weights.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = transform_test(image).unsqueeze(0)  

        with torch.inference_mode():
            outputs = model(image.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        return jsonify({
            'probabilities': {
                'goose': float(probs[0]),
                'jellyfish': float(probs[1]),
                'snail': float(probs[2])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/inference', methods=['GET'])
def inference():
    try:
        image_paths, labels, _ = load_images(image_folder)

        _, test_paths, _, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels
        )

        test_dataset = CustomDataset(test_paths, test_labels, transform_test)
        test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4, prefetch_factor=2)

        accuracy, precision, recall, f1, confma = evaluate_model(test_loader, model)
        return jsonify({
            'metrics': {
                'accuracy': accuracy.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'confusion_matrix': confma.numpy().tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)