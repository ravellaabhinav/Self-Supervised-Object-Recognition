from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    GPT2TokenizerFast
)

import os

app = Flask(__name__)

# Load BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT-GPT2 model
vit_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to("cuda" if torch.cuda.is_available() else "cpu")
vit_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_blip_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
    outputs = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(outputs[0], skip_special_tokens=True)

def generate_vit_gpt2_caption(image):
    pixel_values = vit_feature_extractor(images=image, return_tensors="pt").pixel_values.to(vit_model.device)
    output_ids = vit_model.generate(pixel_values, max_length=50, num_beams=4)
    return vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400
    
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400
    
    try:
        image = Image.open(file.stream).convert("RGB")
        blip_caption = generate_blip_caption(image)
        vit_caption = generate_vit_gpt2_caption(image)

        # Save uploaded file
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(save_path)

        return render_template(
            "result.html",
            blip_caption=blip_caption,
            vit_caption=vit_caption,
            filename=file.filename
        )
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)