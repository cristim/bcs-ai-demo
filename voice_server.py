#!/usr/bin/env python3
"""
Persistent voice cloning server for SpongeBob AI demo
Keeps model in memory for fast generation
"""
from flask import Flask, request, jsonify, send_file
import torch
from TTS.api import TTS
import warnings
import sys
import os
import wave

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

xtts = tts.synthesizer.tts_model
# Computed once: the "voice fingerprint" of the reference sample
_gpt_latent, _speaker_emb = xtts.get_conditioning_latents(audio_path=[VOICE_SAMPLE])
with wave.open(VOICE_SAMPLE, 'rb') as _w:
    SAMPLE_SECONDS = _w.getnframes() / _w.getframerate()

def tokenize(text):
    ids = xtts.tokenizer.encode(text, lang='en')
    pieces = [xtts.tokenizer.tokenizer.id_to_token(i) for i in ids]
    return ids, pieces

@app.route('/voiceprint', methods=['GET'])
def voiceprint():
    emb = _speaker_emb.flatten().cpu().tolist()
    return jsonify({
        'success': True,
        'embedding': [round(v, 4) for v in emb],
        'min': round(min(emb), 4),
        'max': round(max(emb), 4),
        'gpt_cond_latent_shape': list(_gpt_latent.shape),
        'sample_seconds': round(SAMPLE_SECONDS, 2)
    })

@app.route('/reference-sample', methods=['GET'])
def reference_sample():
    return send_file(os.path.abspath(VOICE_SAMPLE), mimetype='audio/wav')

@app.route('/tokenize', methods=['POST'])
def tokenize_route():
    text = (request.json or {}).get('text', '')
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    try:
        ids, pieces = tokenize(text)
        return jsonify({'success': True, 'token_ids': ids, 'tokens': pieces})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

        ids, pieces = tokenize(text)
        with wave.open(output_path, 'rb') as w:
            num_samples, sample_rate = w.getnframes(), w.getframerate()

        return jsonify({
            'success': True,
            'output_file': output_path,
            'generation_time': elapsed,
            'token_ids': ids,
            'tokens': pieces,
            'sample_rate': sample_rate,
            'num_samples': num_samples,
            'duration': num_samples / sample_rate
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
