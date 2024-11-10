# app.py

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import random
import re
import traceback
import os
import requests
from typing import Optional

# Import AI features and session manager
from .ai_features import process_legal_query
from .session_manager import session_manager

# Initialize the app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ChatbotApp")

# Keep your existing patterns
GRATITUDE_PATTERNS = [
    r'thank(?:s| you)',
    r'appreciate it',
    r'grateful',
    r'helpful',
    r'thanks'
]

GOODBYE_PATTERNS = [
    r'bye',
    r'goodbye',
    r'see you',
    r'talk to you later',
    r'exit'
]

CAPABILITY_PATTERNS = [
    r'what can you do',
    r'help me with',
    r'capabilities',
    r'features',
    r'how do you work'
]

GREETING_PATTERNS = [
    r'\b(hi|hey|hello|howdy)\b',
    r'good\s*(morning|afternoon|evening)',
    r'greetings'
]

# Keep your existing responses
GREETING_RESPONSES = [
    "Hello! I'm here to help with your Indian Penal Code related questions. What would you like to know?",
    "Hi! How can I assist you with your IPC-related questions today?",
    "Hello! I'm ready to help you understand the Indian Penal Code better. What's your question?"
]

GRATITUDE_RESPONSES = [
    "You're welcome! Feel free to ask any other questions about the IPC.",
    "Happy to help! Let me know if you need other legal information.",
    "Glad I could assist! Don't hesitate to ask more questions about Indian law."
]

GOODBYE_RESPONSES = [
    "Goodbye! Feel free to return if you have more questions about the IPC.",
    "Take care! I'm here 24/7 if you need more legal assistance.",
]

CAPABILITY_RESPONSES = [
    "I can help you understand various sections of the Indian Penal Code, explain legal terms, and provide information about specific offenses and their punishments.",
    "I'm specialized in the Indian Penal Code. I can explain different sections, help you understand legal concepts, and provide information about criminal laws in India.",
]

def detect_input_type(text: str) -> str:
    """Detect the type of user input."""
    text = text.lower().strip()

    if any(re.search(pattern, text) for pattern in GREETING_PATTERNS):
        return "greeting"
    if any(re.search(pattern, text) for pattern in GRATITUDE_PATTERNS):
        return "gratitude"
    if any(re.search(pattern, text) for pattern in GOODBYE_PATTERNS):
        return "goodbye"
    if any(re.search(pattern, text) for pattern in CAPABILITY_PATTERNS):
        return "capability"
    if len(text.split()) < 2 and not any(re.search(pattern, text) for pattern in GREETING_PATTERNS):
        return "clarification_needed"
    return "legal_query"

@app.post("/chat")
async def chat(request: Request, x_session_id: Optional[str] = Header(None)):
    try:
        # Get and validate input
        data = await request.json()
        user_input = data.get("user_input", "").strip()

        # Get or create session
        session_id = x_session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session()
            logger.info(f"Created new session: {session_id}")
        
        session = session_manager.get_session(session_id)
        
        if not user_input:
            return {
                "response": "I couldn't understand that. Could you please try again?",
                "suggestions": ["Tell me about IPC", "Show common sections", "Explain legal terms"],
                "typing_duration": 500,
                "session_id": session_id
            }

        logger.info(f"Processing chat for session {session_id}: {user_input}")

        # Detect input type and log it
        input_type = detect_input_type(user_input)
        logger.info(f"Detected input type: {input_type}")

        # Initialize response parameters
        response = ""
        suggestions = []
        typing_duration = 1000

        # Handle different input types
        if input_type == "greeting":
            response = random.choice(GREETING_RESPONSES)
            suggestions = [
                "What is Section 302 IPC?",
                "Explain criminal conspiracy",
                "Show punishment for theft"
            ]
            typing_duration = 500

        elif input_type == "clarification_needed":
            response = "Could you please provide more details? I can help you with:"
            response += "\n• Specific IPC sections and their interpretations"
            response += "\n• Legal terms and definitions"
            response += "\n• Criminal offenses and their punishments"
            suggestions = [
                "Show all IPC sections",
                "Common criminal offenses",
                "Basic legal terms"
            ]
            typing_duration = 800

        elif input_type == "gratitude":
            response = random.choice(GRATITUDE_RESPONSES)
            suggestions = [
                "Tell me about another section",
                "Explain more legal terms",
                "Show related provisions"
            ]
            typing_duration = 500

        elif input_type == "goodbye":
            response = random.choice(GOODBYE_RESPONSES)
            suggestions = [
                "Ask another question",
                "Learn about IPC",
                "View legal guidelines"
            ]
            typing_duration = 500
            # End session on goodbye
            session_manager.end_session(session_id)
            session_id = session_manager.create_session()

        elif input_type == "capability":
            response = random.choice(CAPABILITY_RESPONSES)
            suggestions = [
                "Show an example section",
                "List common offenses",
                "Explain IPC structure"
            ]
            typing_duration = 800

        else:
            # Use the AI feature to process legal queries with session memory
            response = process_legal_query(user_input, memory=session.memory)

            # Generate contextual suggestions based on query
            if "section" in user_input.lower():
                suggestions = [
                    "What's the punishment?",
                    "Show related sections",
                    "Explain in simple terms"
                ]
            elif "punishment" in user_input.lower():
                suggestions = [
                    "Show maximum penalty",
                    "Related offenses",
                    "Recent amendments"
                ]
            else:
                suggestions = [
                    "Tell me more",
                    "Show legal provisions",
                    "Practical examples"
                ]

            # Adjust typing duration based on response length
            typing_duration = min(len(response.split()) * 100, 3000)

        # Log the response and return
        logger.info(f"Response type: {input_type}, Response length: {len(response)}")
        return {
            "response": response,
            "suggestions": suggestions,
            "typing_duration": typing_duration,
            "session_id": session_id
        }

    except Exception as e:
        # Handle general errors
        logger.error(f"Unexpected error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        # Create new session on error
        new_session_id = session_manager.create_session()
        return {
            "response": "I apologize, but I'm having trouble processing requests right now. Please try again in a moment.",
            "suggestions": ["Refresh and try again", "Ask about IPC", "Show legal terms"],
            "typing_duration": 500,
            "session_id": new_session_id
        }

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """Endpoint to explicitly end a session."""
    try:
        session_manager.end_session(session_id)
        return {"message": "Session ended successfully"}
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to end session")

@app.get("/session/status/{session_id}")
async def get_session_status(session_id: str):
    """Endpoint to check if a session is valid."""
    try:
        session = session_manager.get_session(session_id)
        return {
            "valid": session is not None,
            "new_session_id": None if session else session_manager.create_session()
        }
    except Exception as e:
        logger.error(f"Error checking session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check session status")

def fetch_pdf_from_drive():
    """Download the PDF from a Google Drive link set in an environment variable."""
    drive_url = os.getenv("PDF_DRIVE_URL")
    if not drive_url:
        raise ValueError("PDF_DRIVE_URL is not set in the environment variables.")

    response = requests.get(drive_url)
    if response.status_code == 200:
        with open("document.pdf", "wb") as f:
            f.write(response.content)
        logger.info("PDF downloaded successfully from Google Drive.")
    else:
        logger.error("Failed to download the PDF from Google Drive.")
        raise Exception("Could not fetch PDF.")