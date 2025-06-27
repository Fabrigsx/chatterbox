from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import torchaudio
from src.chatterbox.models import ChatterboxTTS

app = FastAPI()
tts = ChatterboxTTS()

class TTSRequest(BaseModel):
    text: str
    speaker: str = "default"

@app.post("/speak")
async def speak(req: TTSRequest):
    wav = tts.infer(text=req.text, speaker=req.speaker)
    path = f"output/{req.speaker}.wav"
    torchaudio.save(path, wav.squeeze(0).cpu(), 24000)
    return {"url": f"/audio/{req.speaker}.wav"}

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    from fastapi.responses import FileResponse
    return FileResponse(f"output/{filename}")
