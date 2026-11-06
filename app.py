from flask import Flask, render_template, request, jsonify, send_file
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
# Using gemini-2.5-flash (stable) instead of 2.0-flash-exp for better rate limits
model = genai.GenerativeModel('gemini-2.5-flash')

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
- You often laugh with "Ahahaha!"

SPECIAL CONTEXT: You're talking to students at Berlin Cosmopolitan School (BCS)!
- BCS is an awesome international school in Berlin, Germany with 700 students from 45+ countries
- It's an IB World School with focus on sciences, music, and dance
- Located in Berlin-Mitte near Alexanderplatz
- You think learning is as fun as making Krabby Patties!
- You're excited about schools that celebrate diversity and creativity

IMPORTANT: Keep responses SHORT - maximum 3 sentences (under 30 words total). Be concise and enthusiastic!

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

        total_time = time.time() - start_time

        print(f"⏱️  TIMING: Gemini text generation: {gemini_time:.2f}s | Total: {total_time:.2f}s")

        return jsonify({
            'success': True,
            'response': bot_response,
            'character': char_config['name'],
            'personality_prompt': personality_prompt,
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
