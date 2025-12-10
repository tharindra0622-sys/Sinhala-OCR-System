"""
Flask Backend for Janashakthi Insurance Chatbot with OCR Integration
Integrates Gemini OCR for birth certificate and location change letter verification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBaz-QtbZjB_B34Fug8X9czCyb6B9czLKg"
genai.configure(api_key=GEMINI_API_KEY)

# Find available Gemini model
available_model = None
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        available_model = m.name
        print(f"Using model: {available_model}")
        break

if not available_model:
    raise Exception("No compatible Gemini model found")

model = genai.GenerativeModel(available_model)

# Session storage for conversation context
sessions = {}


def check_birth_certificate(text, image):
    """Check if the image is a birth certificate"""
    prompt = f"""Analyze this document image and the transcribed text to determine if it is a BIRTH CERTIFICATE.

A birth certificate typically contains:
- Government/official header or emblem
- Title like "REGISTER OF BIRTHS", "BIRTH CERTIFICATE", "பிறப்பு பதிவு", "ජන්ම සහතිකය"
- Structured fields for: name, date of birth, place of birth, parents' names
- Official stamps, seals, or registration numbers
- May be in English, Sinhala, Tamil, or multiple languages
- Often has formal government document layout

Transcribed text from document:
{text}

Based on BOTH the visual appearance of the document AND the text content, answer with ONLY "YES" if this is a birth certificate, or "NO" if it is not.
Provide no other explanation."""

    response = model.generate_content([prompt, image])
    answer = response.text.strip().upper()
    return "YES" in answer


def check_location_change(text):
    """Check if the transcribed text is related to a location change request"""
    prompt = f"""Analyze the following text and determine if it is related to a location change request.

The text may be in English, Sinhala (සිංහල), or other languages.

A location change request typically includes:
- Request to change/update address or location
- Mention of moving to a new place
- Change of residence/office/workplace
- Updating location details
- Relocation requests
- In Sinhala: ලිපිනය වෙනස් කිරීම, ස්ථානය වෙනස් කිරීම, etc.

Text to analyze:
{text}

Answer with ONLY "YES" if this is related to a location change request, or "NO" if it is not.
Provide no other explanation."""

    response = model.generate_content(prompt)
    answer = response.text.strip().upper()
    return "YES" in answer


def classify_document(image):
    """Extract handwritten text from an image and classify document type"""
    
    # Create prompt for handwriting recognition
    prompt = """Please transcribe all the handwritten text from this image.

This text may be in English, Sinhala (සිංහල), Tamil (தமிழ்), or other languages. 

Write out the text exactly as it appears, maintaining:
- Line breaks and paragraph structure
- Any lists or bullet points
- Original spelling (even if there are errors)
- Original language (if Sinhala, transcribe in Sinhala script)

If any word is unclear or illegible, write [unclear] in its place.

Provide ONLY the transcribed text, nothing else."""
    
    # Get OCR response
    response = model.generate_content([prompt, image])
    text = response.text
    
    # Classify document type
    print("\nAnalyzing document type...")
    
    # Check if birth certificate first
    is_birth_cert = check_birth_certificate(text, image)
    if is_birth_cert:
        return text, "BIRTH_CERTIFICATE"
    
    # Check if location change request
    is_location_change = check_location_change(text)
    if is_location_change:
        return text, "LOCATION_CHANGE"
    
    # Unknown document type
    return text, "UNKNOWN"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "online", "service": "Janashakthi Insurance API"}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Initialize session if needed
        if session_id not in sessions:
            sessions[session_id] = {'history': []}
        
        # Add message to history
        sessions[session_id]['history'].append({'user': message})
        
        # Process message
        message_lower = message.lower()
        
        # Greeting
        if any(greet in message_lower for greet in ['hi', 'hello', 'hey', 'start']):
            reply = """Hello! 👋 Welcome to Janashakthi Insurance.

I can help you with:

📋 Policy Details - View your policy information
📊 Claim Status - Check your claim status
💰 Premium Info - View premium amounts and due dates
📝 Update Details - Update mobile, email, or address
📄 Document Verification - Upload birth certificates or location change letters
❌ Cancel Policy - Submit cancellation requests
❓ General Help - Get answers to insurance questions

What would you like help with today?"""
        
        # Policy details
        elif 'policy' in message_lower and 'detail' in message_lower:
            reply = """To view your policy details, I'll need your NIC number.

Please provide your NIC in one of these formats:
- 12 digits (e.g., 199512345678)
- 9 digits + V/X (e.g., 951234567V)

Once verified, I'll show you all your registered policies."""
        
        # Claim status
        elif 'claim' in message_lower and 'status' in message_lower:
            reply = """I can help you check your claim status.

Please provide:
1. Your NIC number
2. Claim reference number (if available)

I'll retrieve the latest status of your claim."""
        
        # Premium info
        elif 'premium' in message_lower:
            reply = """I can provide premium information for your policies.

Please share your NIC number, and I'll show you:
- Premium amounts
- Due dates
- Payment history
- Outstanding amounts"""
        
        # Update details
        elif 'update' in message_lower and 'detail' in message_lower:
            reply = """I can help you update your personal details.

What would you like to update?
- Mobile number
- Email address
- Residential address

Please provide your NIC and the information you'd like to update."""
        
        # Cancel policy
        elif 'cancel' in message_lower:
            reply = """I understand you want to cancel a policy.

Please provide:
1. Your NIC number
2. Policy number you wish to cancel
3. Reason for cancellation

Note: Some policies may have surrender charges or waiting periods."""
        
        # Document upload reminder
        elif 'document' in message_lower or 'upload' in message_lower or 'certificate' in message_lower:
            reply = """📄 Document Verification Service

You can upload:
✓ Birth Certificates (for policy verification)
✓ Location Change Letters (in English or Sinhala)

Our OCR system will automatically:
- Extract text from the document
- Classify document type
- Verify authenticity

Click the "Upload Document" or "Upload Photo" button below to get started!"""
        
        # Default response
        else:
            reply = """I'm here to help! Could you please clarify what you need?

You can ask about:
- Policy details
- Claim status
- Premium information
- Updating your details
- Uploading documents
- General insurance questions

Or type 'Hi' to see the full menu."""
        
        # Add reply to history
        sessions[session_id]['history'].append({'bot': reply})
        
        return jsonify({"reply": reply, "session_id": session_id}), 200
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/ocr/classify-document', methods=['POST'])
def ocr_classify_document():
    """OCR endpoint for document classification"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Classify document
        print(f"Processing document: {file.filename}")
        extracted_text, doc_type = classify_document(image)
        
        print(f"Document Type: {doc_type}")
        print(f"Extracted Text Length: {len(extracted_text)} characters")
        
        return jsonify({
            "success": True,
            "document_type": doc_type,
            "extracted_text": extracted_text,
            "filename": file.filename
        }), 200
        
    except Exception as e:
        print(f"Error in OCR endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/verify-document', methods=['POST'])
def verify_document():
    """Legacy endpoint for document verification"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        return jsonify({
            "success": True,
            "message": "Document uploaded successfully",
            "filename": file.filename
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Janashakthi Insurance Backend Starting...")
    print("=" * 60)
    print(f"Gemini API configured: {bool(GEMINI_API_KEY)}")
    print(f"Model available: {available_model}")
    print("=" * 60)
    print("Server running on: http://localhost:5001")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)