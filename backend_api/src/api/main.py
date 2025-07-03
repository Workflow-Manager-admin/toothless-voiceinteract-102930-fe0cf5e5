"""
FastAPI backend API for Toothless VoiceInteract: 
Provides REST endpoints for AI chat and TTS, proxying to Gemini Pro and ElevenLabs, respectively.

- POST /chat: Forwards prompt to Gemini Pro and returns AI text response.
- POST /tts: Forwards text to ElevenLabs for speech synthesis, returns audio bytes.

Easily extensible for more providers or endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx
import os
from typing import Optional

# ---- Secure Configuration Loader ----

def get_env_var_securely(var_name: str, default: Optional[str] = "", required: bool = False):
    """Fetch env variable or raise HTTP 500 if required and missing."""
    v = os.environ.get(var_name, default)
    if required and not v:
        raise HTTPException(status_code=500, detail=f"{var_name} not configured.")
    return v

# Provide default endpoints, keys are loaded dynamically in endpoints for max security
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent")
ELEVENLABS_ENDPOINT = os.environ.get("ELEVENLABS_ENDPOINT", "https://api.elevenlabs.io/v1/text-to-speech")

# ---- FastAPI App Initialization ----

app = FastAPI(
    title="Toothless VoiceInteract Backend API",
    version="1.0.0",
    description="""
Backend REST API for chat (Gemini Pro) and TTS (ElevenLabs) proxying.
- POST `/chat`: Gemini Pro AI chat.
- POST `/tts`: ElevenLabs speech synthesis.
"""
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Pydantic Models ----

class ChatRequest(BaseModel):
    # PUBLIC_INTERFACE
    prompt: str = Field(..., description="User's input text prompt to AI.")

class ChatResponse(BaseModel):
    # PUBLIC_INTERFACE
    reply: str = Field(..., description="AI's response to the prompt.")

class TTSRequest(BaseModel):
    # PUBLIC_INTERFACE
    text: str = Field(..., description="Text to synthesize as speech.")
    voice_id: Optional[str] = Field(None, description="Optional ElevenLabs voice ID for synthesis.")

class TTSResponse(BaseModel):
    # PUBLIC_INTERFACE
    audio_content: str = Field(..., description="Base64-encoded speech audio.")

# ---- Root Health Endpoint ----

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# ---- AI Chat Endpoint ----

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["AI", "Chat"],
    summary="AI chat via Gemini Pro",
    description="Posts a prompt to Gemini Pro via Google AI API and returns the AI's reply.",
)
async def chat(request: ChatRequest):
    """
    PUBLIC_INTERFACE
    Receives a chat prompt and returns Gemini Pro's response.

    - Accepts: { "prompt": "your input" }
    - Returns: { "reply": "AI reply" }
    """
    # Securely load Gemini Pro API key each time
    gemini_api_key = get_env_var_securely("GEMINI_API_KEY", required=True)

    payload = {
        "contents": [
            {"parts": [{"text": request.prompt}]}
        ]
    }
    # The Google Gemini generative language API expects ?key= in query
    # Do not log the key!
    headers = {
        "Authorization": f"Bearer {gemini_api_key}",
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{GEMINI_ENDPOINT}?key={gemini_api_key}",
                json=payload,
                headers=headers,
            )
        if r.status_code != 200:
            # scrub any key occurrence in response text
            msg = r.text.replace(gemini_api_key, "[secure]")
            raise HTTPException(status_code=502, detail=f"Gemini error: {msg}")
        result = r.json()
        reply = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )
        if not reply:
            raise HTTPException(status_code=502, detail="No chat response from Gemini Pro.")
        return ChatResponse(reply=reply)
    except httpx.RequestError:
        # Do not leak secrets
        raise HTTPException(status_code=502, detail="Error communicating with Gemini Pro.")

# ---- Text-to-Speech (TTS) Endpoint ----

@app.post(
    "/speak",
    tags=["TTS", "Voice"],
    summary="Text-to-speech via ElevenLabs",
    description="Forwards text to ElevenLabs and returns generated speech audio (as streaming audio/mpeg).",
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "Audio stream (MPEG)",
        }
    }
)
async def speak(request: TTSRequest):
    """
    PUBLIC_INTERFACE
    Receives text and returns ElevenLabs synthesized speech audio.

    - Accepts: { "text": "Hello world", "voice_id": "optional_id" }
    - Returns: audio/mpeg stream (use as file download or direct audio playback)
    """
    # Load API key and voice ID at request time for best env hygiene
    eleven_api_key = get_env_var_securely("ELEVEN_API_KEY", required=True)
    # Voice ID: allow override, else pick from env
    eleven_voice_id = request.voice_id or get_env_var_securely("ELEVEN_VOICE_ID", required=True)
    endpoint = f"{ELEVENLABS_ENDPOINT}/{eleven_voice_id}"

    payload = {
        "text": request.text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }
    headers = {
        "xi-api-key": eleven_api_key,
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                endpoint,
                headers=headers,
                json=payload
            )
        if resp.status_code != 200:
            msg = resp.text.replace(eleven_api_key, "[secure]").replace(eleven_voice_id, "[secure]")
            raise HTTPException(status_code=502, detail=f"ElevenLabs error: {msg}")
        # Stream audio response directly
        return StreamingResponse(resp.aiter_bytes(), media_type="audio/mpeg")
    except httpx.RequestError:
        # Don't reveal any sensitive error details
        raise HTTPException(status_code=502, detail="Error communicating with ElevenLabs.")

# ---- API Docs route for WebSocket/Realtime connection help (for extensibility) ----

@app.get("/docs/ws_help", tags=["Docs"])
def websocket_usage():
    """
    PUBLIC_INTERFACE
    Returns usage information for future WebSocket/realtime API extensions.
    """
    return {
        "usage": "This API currently supports only REST endpoints. For realtime or websocket-based features (like streaming chat or TTS), use future endpoints at /ws/* with appropriate socket protocols."
    }

# ---- Deprecated /tts route for backward compatibility, wraps /speak ----
@app.post("/tts", tags=["TTS", "Voice"], include_in_schema=False)
async def tts_alias(request: TTSRequest):
    """Deprecated. Forwards request to /speak for legacy clients."""
    return await speak(request)

# ---- Multipurpose Error Handler ----
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Do not leak secrets in uncaught exceptions - just a generic error
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
