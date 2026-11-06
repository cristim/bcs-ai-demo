#!/usr/bin/env python3
"""
Persistent voice cloning server for SpongeBob AI demo
Keeps model in memory for fast generation
"""
from flask import Flask, request, jsonify, send_file, Response
import torch
from TTS.api import TTS
import warnings
import sys
import os
import wave
import numpy as np
import soundfile as sf
import threading
import time

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

# XTTS's stream generator passes int token ids where the installed transformers expects tensors
from transformers.generation.utils import GenerationMixin
_orig_attention_mask = GenerationMixin._prepare_attention_mask_for_generation
def _attention_mask_with_tensor_ids(self, inputs, pad_token_id, eos_token_id):
    as_tensor = lambda v: v if v is None or torch.is_tensor(v) else torch.as_tensor(v, device=inputs.device)
    return _orig_attention_mask(self, inputs, as_tensor(pad_token_id), as_tensor(eos_token_id))
GenerationMixin._prepare_attention_mask_for_generation = _attention_mask_with_tensor_ids
_orig_logits_processor = GenerationMixin._get_logits_processor
def _logits_processor_with_special_tokens(self, generation_config, *args, **kwargs):
    if not hasattr(generation_config, '_eos_token_tensor'):
        self._prepare_special_tokens(generation_config, True, device=self.device)
    return _orig_logits_processor(self, generation_config, *args, **kwargs)
GenerationMixin._get_logits_processor = _logits_processor_with_special_tokens
_orig_update_kwargs = GenerationMixin._update_model_kwargs_for_generation
def _update_kwargs_with_cache_position(self, outputs, model_kwargs, *args, **kwargs):
    model_kwargs.setdefault('cache_position', torch.zeros(1, dtype=torch.long))  # unused by XTTS's GPT
    return _orig_update_kwargs(self, outputs, model_kwargs, *args, **kwargs)
GenerationMixin._update_model_kwargs_for_generation = _update_kwargs_with_cache_position
# Newer transformers fold the temperature/top-k/top-p warpers into _get_logits_processor
from transformers import LogitsProcessorList
GenerationMixin._get_logits_warper = lambda self, generation_config, *a, **k: LogitsProcessorList()

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

# The model is not thread-safe: one generation at a time
_gen_lock = threading.Lock()
# Metadata of streamed generations, keyed by output filename
_stream_info = {}

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
    try:
        data = request.json
        text = data.get('text', '')
        output_path = data.get('output', 'output.wav')

        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        with _gen_lock:
            start_time = time.time()
            # Low temperature keeps the cartoon voice closest to the reference (A/B tested)
            out = xtts.inference(
                text, "en", _gpt_latent, _speaker_emb,
                temperature=0.3, top_p=0.8, repetition_penalty=10.0, enable_text_splitting=True
            )
        sf.write(output_path, np.asarray(out["wav"], dtype=np.float32), xtts.config.audio.output_sample_rate)

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

@app.route('/generate-stream', methods=['POST'])
def generate_stream():
    """Stream int16 PCM chunks as XTTS produces them; save the full WAV at the end"""
    data = request.json or {}
    text = data.get('text', '')
    output_path = data.get('output', 'output.wav')
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    sample_rate = xtts.config.audio.output_sample_rate

    def stream():
        with _gen_lock:
            start = time.time()
            first_chunk = None
            chunks = []
            for chunk in xtts.inference_stream(
                text, "en", _gpt_latent, _speaker_emb, stream_chunk_size=20,
                temperature=0.3, top_p=0.8, repetition_penalty=10.0, enable_text_splitting=True
            ):
                if first_chunk is None:
                    first_chunk = time.time() - start
                wav = np.clip(chunk.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
                chunks.append(wav)
                yield (wav * 32767).astype('<i2').tobytes()
            elapsed = time.time() - start
            full = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            sf.write(output_path, full, sample_rate)
            ids, pieces = tokenize(text)
            _stream_info[os.path.basename(output_path)] = {
                'success': True,
                'generation_time': elapsed,
                'first_chunk_time': first_chunk,
                'token_ids': ids,
                'tokens': pieces,
                'sample_rate': sample_rate,
                'num_samples': len(full),
                'duration': len(full) / sample_rate
            }
            print(f"Streamed voice: first chunk {first_chunk:.2f}s, total {elapsed:.2f}s", file=sys.stderr)

    return Response(stream(), mimetype='application/octet-stream',
                    headers={'X-Sample-Rate': str(sample_rate)})

@app.route('/generate-info', methods=['GET'])
def generate_info():
    info = _stream_info.get(os.path.basename(request.args.get('output', '')))
    if not info:
        return jsonify({'success': False, 'error': 'No info for that output (still generating?)'}), 404
    return jsonify(info)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ready', 'device': device})

if __name__ == '__main__':
    # Run on different port to avoid conflicts
    app.run(host='127.0.0.1', port=5001, debug=False)
