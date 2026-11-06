#!/usr/bin/env python3
"""
Persistent voice cloning server for SpongeBob AI demo
Keeps model in memory for fast generation
"""
from flask import Flask, request, jsonify
import torch
from TTS.api import TTS
import warnings
import sys
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ weights_only issue
import torch.serialization
torch.serialization.add_safe_globals(['TTS.tts.configs.xtts_config.XttsConfig'])

# Monkey patch torch.load
original_torch_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_load

app = Flask(__name__)

# Initialize TTS model once at startup
print("Loading XTTS model...", file=sys.stderr)

# Use Metal GPU (MPS) if available for 49% faster generation
# After warmup: MPS ~1.8s vs CPU ~3.5s per generation
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Metal GPU (MPS) for acceleration", file=sys.stderr)
else:
    device = "cpu"
    print("Using CPU (Metal GPU not available)", file=sys.stderr)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"Model loaded on {device} and ready!", file=sys.stderr)

# Voice sample path (6-30 seconds optimal for XTTS-v2)
VOICE_SAMPLE = os.getenv("VOICE_SAMPLE", "spongebob-sample-with-laugh.wav")  # Original + 3s laugh sound effect
if not os.path.isfile(VOICE_SAMPLE):
    sys.exit(f"Voice sample not found: {VOICE_SAMPLE} (it is gitignored; provide it or set VOICE_SAMPLE)")

@app.route('/generate', methods=['POST'])
def generate():
    """Generate speech with cloned voice"""
    import time
    try:
        data = request.json
        text = data.get('text', '')
        output_path = data.get('output', 'output.wav')

        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        start_time = time.time()

        tts.tts_to_file(
            text=text,
            speaker_wav=VOICE_SAMPLE,
            language="en",
            file_path=output_path
        )

        elapsed = time.time() - start_time
        print(f"Voice generation took {elapsed:.2f} seconds", file=sys.stderr)

        return jsonify({
            'success': True,
            'output_file': output_path,
            'generation_time': elapsed
        })

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ready', 'device': device})

if __name__ == '__main__':
    # Run on different port to avoid conflicts
    app.run(host='127.0.0.1', port=5001, debug=False)
