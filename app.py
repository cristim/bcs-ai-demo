from flask import Flask, render_template, request, jsonify, send_file, Response
import google.generativeai as genai
import os
import subprocess
import json
import uuid
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# gemini-2.5-flash is closed to new API keys; the flash-lite alias answers in ~1s and has free-tier quota
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-flash-lite-latest')
model = genai.GenerativeModel(GEMINI_MODEL)

# Character prompts - easy to add more characters later!
CHARACTERS = {
    'spongebob': {
        'name': 'SpongeBob SquarePants',
        'prompt': """You are SpongeBob SquarePants from the cartoon series. You are:
- Always optimistic, cheerful, and enthusiastic
- You say "I'm ready!" often when excited
- You love your job at the Krusty Krab making Krabby Patties
- Your best friend is Patrick Star
- You love jellyfishing and bubble blowing
- You're loyal, kind, and sometimes naive

SPECIAL CONTEXT: You're talking to students at Berlin Cosmopolitan School (BCS)!
- BCS is an awesome international school in Berlin, Germany
- It's an IB World School with focus on sciences, music, and dance
- Located in Berlin-Mitte near Alexanderplatz
- You think learning is as fun as making Krabby Patties!
- You're excited about schools that celebrate diversity and creativity

IMPORTANT: Keep responses VERY SHORT - one or two short sentences, 15 words maximum. Be enthusiastic! Never write out laughter (no "Ahahaha", "haha" or similar).

User's message: """
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/presenter')
def presenter():
    return render_template('presenter.html')

@app.route('/chat', methods=['POST'])
def chat():
    import time
    personality_prompt = None  # also reported on the error path
    try:
        start_time = time.time()

        data = request.json
        user_message = data.get('message', '')
        character = data.get('character', 'spongebob')

        # Get character prompt
        char_config = CHARACTERS.get(character, CHARACTERS['spongebob'])
        full_prompt = char_config['prompt'] + user_message
        personality_prompt = char_config['prompt'].rsplit("User's message:", 1)[0].strip()

        # Generate response using Gemini
        gemini_start = time.time()
        response = model.generate_content(full_prompt)
        bot_response = response.text
        gemini_time = time.time() - gemini_start
        try:
            token_counts = {
                'prompt': model.count_tokens(full_prompt).total_tokens,
                'answer': model.count_tokens(bot_response).total_tokens
            }
        except Exception:
            token_counts = None

        total_time = time.time() - start_time

        print(f"⏱️  TIMING: Gemini text generation: {gemini_time:.2f}s | Total: {total_time:.2f}s")

        return jsonify({
            'success': True,
            'response': bot_response,
            'character': char_config['name'],
            'personality_prompt': personality_prompt,
            'token_counts': token_counts,
            'timing': {
                'gemini': gemini_time,
                'total': total_time
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'personality_prompt': personality_prompt
        }), 500

@app.route('/generate-voice', methods=['POST'])
def generate_voice():
    """Generate cloned voice audio for given text"""
    import time
    try:
        start_time = time.time()

        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        # Generate unique filename
        audio_id = str(uuid.uuid4())
        output_file = f"static/audio/{audio_id}.wav"

        # Ensure audio directory exists
        os.makedirs('static/audio', exist_ok=True)

        # Call persistent voice server via HTTP
        http_start = time.time()
        response = requests.post(
            'http://127.0.0.1:5001/generate',
            json={'text': text, 'output': output_file},
            timeout=120  # first MPS generation includes ~60s of kernel warmup
        )
        http_time = time.time() - http_start

        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('success'):
                total_time = time.time() - start_time
                voice_gen_time = response_data.get('generation_time', 0)
                network_overhead = http_time - voice_gen_time

                print(f"🎙️  TIMING: Voice generation: {voice_gen_time:.2f}s | Network: {network_overhead:.2f}s | Total: {total_time:.2f}s")

                return jsonify({
                    'success': True,
                    'audio_url': f'/static/audio/{audio_id}.wav',
                    'token_ids': response_data.get('token_ids'),
                    'tokens': response_data.get('tokens'),
                    'sample_rate': response_data.get('sample_rate'),
                    'num_samples': response_data.get('num_samples'),
                    'duration': response_data.get('duration'),
                    'timing': {
                        'voice_generation': voice_gen_time,
                        'network_overhead': network_overhead,
                        'total': total_time
                    }
                })

        return jsonify({
            'success': False,
            'error': 'Voice generation failed'
        }), 500

    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': 'Voice server not running. Start it with: venv-voice/bin/python voice_server.py'
        }), 500
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Voice generation timeout'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

VOICE_SERVER = 'http://127.0.0.1:5001'
VOICE_OFFLINE = {'success': False, 'error': 'voice server offline'}

@app.route('/generate-voice-stream', methods=['POST'])
def generate_voice_stream():
    """Stream int16 PCM of the cloned voice as it is generated; the WAV is saved at X-Audio-Url"""
    text = (request.json or {}).get('text', '')
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    audio_id = str(uuid.uuid4())
    os.makedirs('static/audio', exist_ok=True)
    try:
        upstream = requests.post(
            f'{VOICE_SERVER}/generate-stream',
            json={'text': text, 'output': f'static/audio/{audio_id}.wav'},
            stream=True, timeout=(5, 120)
        )
    except requests.exceptions.RequestException:
        return jsonify(VOICE_OFFLINE), 503
    if upstream.status_code != 200:
        upstream.close()
        return jsonify({'success': False, 'error': 'Voice generation failed'}), 500

    def relay():
        try:
            yield from upstream.iter_content(chunk_size=None)
        finally:
            upstream.close()

    return Response(relay(), mimetype='application/octet-stream', headers={
        'X-Audio-Url': f'/static/audio/{audio_id}.wav',
        'X-Sample-Rate': upstream.headers.get('X-Sample-Rate', '24000'),
        'Cache-Control': 'no-store',
        'X-Accel-Buffering': 'no'
    })

@app.route('/voice-info')
def voice_info():
    """Metadata of a streamed generation, shaped like /generate-voice's response"""
    audio_url = request.args.get('audio_url', '')
    try:
        r = requests.get(f'{VOICE_SERVER}/generate-info', params={'output': os.path.basename(audio_url)}, timeout=5)
    except requests.exceptions.RequestException:
        return jsonify(VOICE_OFFLINE), 503
    info = r.json() if r.status_code == 200 else None
    if not info or not info.get('success'):
        return jsonify({'success': False, 'error': 'Voice info not available'}), 404
    gen = info['generation_time']
    return jsonify({
        'success': True,
        'audio_url': audio_url,
        'token_ids': info['token_ids'],
        'tokens': info['tokens'],
        'sample_rate': info['sample_rate'],
        'num_samples': info['num_samples'],
        'duration': info['duration'],
        'timing': {
            'voice_generation': gen,
            'first_chunk': info['first_chunk_time'],
            'total': gen,
            'network_overhead': 0
        }
    })

def voice_proxy(method, path, **kwargs):
    try:
        r = requests.request(method, VOICE_SERVER + path, timeout=5, **kwargs)
        return r.content, r.status_code, {'Content-Type': r.headers.get('Content-Type', 'application/json')}
    except requests.exceptions.RequestException:
        return jsonify(VOICE_OFFLINE), 503

@app.route('/voiceprint')
def voiceprint():
    return voice_proxy('GET', '/voiceprint')

@app.route('/reference-sample')
def reference_sample():
    return voice_proxy('GET', '/reference-sample')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    return voice_proxy('POST', '/tokenize', json=request.json)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
