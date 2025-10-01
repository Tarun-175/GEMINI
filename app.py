from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import os
from google import genai
from google.genai import types
import json

app = Flask(__name__)
CORS(app)

# Initialize Gemini client
client = genai.Client(api_key="PUT YOUR API KEY HERE")

@app.route('/')
def index():
    return render_template('simple_translator.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Light validation - only reject if message is empty or too short
    if len(user_message.strip()) < 1:
        return jsonify({'error': 'Please provide text to translate.'}), 400
    
    def generate_response():
        model = "gemini-2.5-pro"
        
        # System prompt to make AI act as a language translator only
        system_prompt = """You are a language translator that ONLY translates English text to Tamil and Spanish. 

STRICT RULES:
1. You ONLY accept English text for translation
2. You MUST provide translations in EXACTLY this format:
   Spanish: [Spanish translation]
   Tamil: [Tamil translation]
3. You MUST NOT respond to any other queries, questions, or requests
4. If the input is not English text to be translated, respond with: "I only translate English text to Tamil and Spanish. Please provide English text to translate."
5. Do not engage in conversations, answer questions, or provide any other assistance
6. Focus solely on accurate translation between English, Tamil, and Spanish

Example:
Input: "how are you?"
Spanish: ¿Cómo estás?
Tamil: எப்படி இருக்கிறாய்?"""

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"{system_prompt}\n\nTranslate this English text: {user_message}"),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig()

        import time
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to translate (attempt {attempt + 1}): {user_message}")
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text:
                        print(f"Received chunk: {chunk.text}")
                        yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return  # Success, exit retry loop
                
            except Exception as e:
                print(f"Error occurred on attempt {attempt + 1}: {str(e)}")
                
                # Check if it's a 503 overload error
                if "503" in str(e) or "overloaded" in str(e).lower():
                    if attempt < max_retries - 1:  # Not the last attempt
                        print(f"Model overloaded, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        error_msg = "The translation service is currently busy. Please try again in a few moments."
                else:
                    error_msg = f"Translation error: {str(e)}"
                
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return
    
    return Response(generate_response(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
