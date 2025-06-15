"""
QuantAI Restaurant API Server
This module provides a FastAPI server that integrates both text and voice processing capabilities
from the QuantAI Restaurant AI Assistant system.
"""

import os
import logging
from typing import Optional
from pathlib import Path
import tempfile
import uuid
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import our agent implementations
from agent import QuantAIAgent
# from voice_agent import VoiceAgent, TextToSpeech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantai_restaurant_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Restaurant API",
    description="API server for QuantAI Restaurant's text and voice processing capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our agents
text_agent = QuantAIAgent()
# voice_agent = VoiceAgent()
# tts_engine = TextToSpeech()

# Create temp directory for audio files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

class TextQuery(BaseModel):
    """Model for text query requests."""
    text: str
    language: Optional[str] = "english"

class TextResponse(BaseModel):
    """Model for text query responses."""
    success: bool
    response: str
    language: str
    detected_language: Optional[str] = None

# class VoiceResponse(BaseModel):
#     """Model for voice query responses."""
#     text: str
#     audio_url: Optional[str] = None
#     error: Optional[str] = None

def cleanup_old_files(directory: Path, max_age_hours: int = 1):
    """Clean up old temporary files."""
    current_time = datetime.now().timestamp()
    for file in directory.glob("*"):
        if current_time - file.stat().st_mtime > (max_age_hours * 3600):
            try:
                file.unlink()
                logger.info(f"Cleaned up old file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup."""
    logger.info("Starting QuantAI Restaurant API Server")
    cleanup_old_files(TEMP_DIR)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on server shutdown."""
    logger.info("Shutting down QuantAI Restaurant API Server")
    cleanup_old_files(TEMP_DIR)

@app.post("/text-query", response_model=TextResponse)
async def process_text_query(query: TextQuery):
    """
    Process a text query and return the response.
    """
    try:
        logger.info(f"Processing text query: {query.text[:100]}...")
        
        # Validate and normalize the requested language
        is_valid, normalized_language = text_agent.language_manager.validate_language(query.language)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {query.language}. Use /languages endpoint to see available languages."
            )
            
        # Set the validated language
        text_agent.user_language = normalized_language
        logger.info(f"Language set to: {normalized_language}")
        
        # Generate response in English first
        response = text_agent.generate_response(query.text)
        
        # Translate response if not English
        if normalized_language.lower() != "english":
            logger.info(f"Translating response to {normalized_language}")
            try:
                response = text_agent.translate_text(response)
                if not response:
                    raise ValueError("Translation failed")
            except Exception as e:
                logger.error(f"Translation error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to translate response to {normalized_language}"
                )
        
        return TextResponse(
            success=True,
            response=response,
            language=normalized_language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text query: {str(e)}"
        )

# @app.post("/voice-query", response_model=VoiceResponse)
# async def process_voice_query(
#     audio_file: UploadFile = File(...),
#     background_tasks: BackgroundTasks = None
# ):
#     """
#     Process a voice query and return both text and synthesized speech response.
#     """
#     try:
#         logger.info(f"Processing voice query from file: {audio_file.filename}")
        
#         # Save uploaded audio to temporary file
#         temp_input = TEMP_DIR / f"input_{uuid.uuid4()}.wav"
#         with open(temp_input, "wb") as buffer:
#             content = await audio_file.read()
#             buffer.write(content)
        
#         # Process voice input
#         text = voice_agent.speech_to_text.convert_to_text(temp_input.read_bytes())
#         if not text:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Could not understand audio input"
#             )
            
#         logger.info(f"Transcribed text: {text}")
        
#         # Generate response using the text agent
#         response_text = text_agent.generate_response(text)
        
#         # Convert response to speech
#         audio_data = tts_engine.convert_to_speech(response_text)
        
#         if not audio_data:
#             # Return text-only response if speech conversion fails
#             return VoiceResponse(
#                 text=response_text,
#                 error="Speech synthesis unavailable"
#             )
            
#         # Save audio response to temporary file
#         temp_output = TEMP_DIR / f"output_{uuid.uuid4()}.mp3"
#         with open(temp_output, "wb") as f:
#             f.write(audio_data)
            
#         # Schedule cleanup of temporary files
#         if background_tasks:
#             background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        
#         # Generate URL for audio file
#         audio_url = f"/audio/{temp_output.name}"
        
#         return VoiceResponse(
#             text=response_text,
#             audio_url=audio_url
#         )
        
#     except Exception as e:
#         logger.error(f"Error processing voice query: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing voice query: {str(e)}"
#         )
#     finally:
#         # Clean up input file
#         if temp_input.exists():
#             temp_input.unlink()

# @app.get("/audio/{filename}")
# async def get_audio(filename: str):
#     """
#     Serve generated audio files.
#     """
#     file_path = TEMP_DIR / filename
#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="Audio file not found")
        
#     return FileResponse(
#         file_path,
#         media_type="audio/mpeg",
#         filename=filename
#     )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/languages")
async def get_available_languages():
    """
    Get list of supported languages.
    """
    return {
        "languages": sorted(list(text_agent.language_manager.supported_languages)),
        "aliases": text_agent.language_manager.language_aliases
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for consistent error responses.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "error": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 