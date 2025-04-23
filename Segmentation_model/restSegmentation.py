from flask import Flask, request, jsonify, render_template, send_file
import torch
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from model import SegmentationModel
from dataset import SegmentationDataset
from performSegmentation import transformImageValid, inference as evaluation,  LABELS, collectPaths, VALIDATION_PATH, transformMask, BATCH_SIZE, PREFETCH_FAC, NUM_WORKERS
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from io import BytesIO
import traceback
import logging


app = Flask(__name__)

model = SegmentationModel()
diceObj = torch.load('./modelData/best_dice.pth')
model.load_state_dict(diceObj['model_weights'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def doOnePicture(imgFile):
    imgFile.seek(0)
    org = Image.open(imgFile).convert('RGB').resize((224,224), Image.BILINEAR)
    img = transformImageValid(org).unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = torch.argmax(model(img), dim=1)

    prediction = prediction.cpu()
    return visualize_prediction_overlay(org, prediction)

def visualize_prediction_overlay(original_img, prediction):

    pred_mask = prediction.squeeze(0).numpy()
    overlay = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
    
    class_colors = {
        0: [0, 0, 0, 0],      
        1: [0, 0, 255, 150],  # blue cello
        2: [0, 255, 0, 150],  # green piano
        3: [255, 0, 0, 150]   # red pizza
    }
    
    for class_idx, color in class_colors.items():
        overlay[pred_mask == class_idx] = color
    
    original_img = original_img.convert('RGBA')
    overlay_img = Image.fromarray(overlay, 'RGBA')
    result = Image.alpha_composite(original_img, overlay_img)
    
    side_by_side = Image.new('RGBA', (224 * 2, 224))
    side_by_side.paste(original_img, (0, 0))
    side_by_side.paste(result, (224, 0))

    img_byte_arr = BytesIO()
    side_by_side.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

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
        image = io.BytesIO(file.read())
        result_image = doOnePicture(image)
        return send_file(
                result_image,
                mimetype='image/png',
                as_attachment=False,
                download_name='result.png'
            )
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
    
@app.route('/inference', methods=['GET'])
def inference():
    try:
        validationSamples = collectPaths(VALIDATION_PATH)
        validationDataset = SegmentationDataset(validationSamples, LABELS, transformImageValid, transformMask, isTraining=False)
        validDl = DataLoader(validationDataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FAC, pin_memory=True)
        metrics = evaluation(validDl, model)
        return jsonify({
            'metrics': {
                'dice': metrics['dice'].item(),
                'f1_macro': metrics['f1_macro'].item(),
                'f1_micro': metrics['f1_micro'].item(),
                'Cello_dice': metrics['dice_single'][1].item(),
                'Piano_dice': metrics['dice_single'][2].item(),
                'Pizza_dice': metrics['dice_single'][3].item()
            }
        })
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)