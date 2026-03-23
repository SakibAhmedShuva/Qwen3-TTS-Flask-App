from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os
import uuid
import torch
import soundfile as sf
from num2words import num2words

# Import the official Qwen3-TTS model
try:
    from qwen_tts import Qwen3TTSModel
    MODEL_AVAILABLE = True
except ImportError:
    print("Warning: qwen-tts is not installed properly. Running in text-only debug mode.")
    MODEL_AVAILABLE = False

app = Flask(__name__, static_folder='static')
CORS(app) # Enable CORS so the frontend can communicate with the API

# The Cyrillic vowel hack to prevent Qwen3 from stuttering
CHAR_REPLACEMENTS = {
    'Ө': 'О',
    'Ү': 'У',
    'ө': 'о',
    'ү': 'у'
}

def process_mongolian_text(text):
    """Normalizes digits to Mongolian words and replaces problematic vowels."""
    
    # 1. Replace minus signs before numbers with the Mongolian word "хасах "
    text = re.sub(r'-\s*(?=\d)', 'хасах ', text)
    
    # 2. Convert digits to Mongolian words using num2words
    def replace_number(match):
        num_str = match.group(0).replace(',', '') # remove commas (e.g. 45,000)
        try:
            num = float(num_str) if '.' in num_str else int(num_str)
            return num2words(num, lang='mn')
        except Exception:
            return num_str # Fallback

    # Matches integers, decimals, and comma-separated large numbers
    text = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+\b', replace_number, text)
    
    # 3. Swap the uniquely Mongolian vowels with standard Russian ones
    # (We do this AFTER number conversion, because words like "дөрөв" (4) contain 'ө')
    for old_char, new_char in CHAR_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)
        
    return text

# Load the Qwen3-TTS Model globally so it only loads into VRAM once at startup
model = None
if MODEL_AVAILABLE:
    print("Loading Qwen3-TTS Model... (This may take a moment)")
    # Using the 0.6B CustomVoice model for speed. Change to 1.7B for higher quality.
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", 
        device_map="auto",
        dtype=torch.float16
    )
    print("Model loaded successfully!")

@app.route('/generate', methods=['POST'])
def generate_audio():
    data = request.json
    raw_text = data.get('text', '')
    
    if not raw_text:
        return jsonify({'error': 'No text provided'}), 400
        
    # Process the text using our custom logic
    processed_text = process_mongolian_text(raw_text)
    
    if not MODEL_AVAILABLE or model is None:
        return jsonify({
            'processed_text': processed_text,
            'audio_url': None,
            'message': 'Text normalized, but Qwen-TTS model failed to load.'
        })

    try:
        # Generate the audio array
        wavs, sr = model.generate_custom_voice(
            text=processed_text,
            language="Auto", # Auto-detects the script
            speaker="Aiden"  # Try "Serena", "Ryan", or other preset voices
        )
        
        # Ensure the static/audio directory exists
        os.makedirs('static/audio', exist_ok=True)
        
        # Save output to a unique .wav file
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join('static/audio', filename)
        sf.write(filepath, wavs[0], sr)
        
        return jsonify({
            'processed_text': processed_text,
            'audio_url': f'/static/audio/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)