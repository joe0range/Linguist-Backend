import os
import json
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Linguist AI Translator — Groq", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"   # fast + free-tier

SUPPORTED_LANGUAGES = [
    "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)",
    "Czech", "Danish", "Dutch", "English", "Finnish", "French",
    "German", "Greek", "Hebrew", "Hindi", "Indonesian", "Italian",
    "Japanese", "Korean", "Malay", "Norwegian", "Persian", "Polish",
    "Portuguese", "Romanian", "Russian", "Spanish", "Swahili",
    "Swedish", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian",
    "Urdu", "Vietnamese",
]


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    source_language: str = Field(default="auto")
    target_language: str


def build_prompt(text: str, source: str, target: str) -> str:
    if source == "auto":
        from_clause = "Detect the source language automatically."
        detect_instruction = (
            'Since you are auto-detecting, prepend exactly one line: '
            '"DETECTED_LANG: <language name>" then a newline, then the translation.'
        )
    else:
        from_clause = f"The source language is {source}."
        detect_instruction = ""

    return (
        f"You are a professional translator. {from_clause} "
        f"Translate the following text to {target}. "
        f"Output ONLY the translated text with no explanations, quotes, preamble, or notes. "
        f"{detect_instruction}\n\nText:\n{text}"
    )


@app.get("/")
async def root():
    return {"status": "ok", "service": "Linguist AI Translator (Groq)"}


@app.get("/languages")
async def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/translate")
async def translate(req: TranslateRequest):
    if req.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {req.target_language}")

    if req.source_language != "auto" and req.source_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {req.source_language}")

    prompt = build_prompt(req.text, req.source_language, req.target_language)

    def stream_translation():
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    payload = json.dumps({"type": "delta", "text": delta})
                    yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        stream_translation(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
