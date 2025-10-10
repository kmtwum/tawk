import asyncio

from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import requests


import os
import torch
import gc
import uuid

app = FastAPI()
from dotenv import load_dotenv

load_dotenv()


def cleanup_files(temp_files: list):
    """Background task to clean up files after a response is sent"""
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    torch.cuda.empty_cache()
    gc.collect()


def generate_tts(text: str, tts_preference: str, out_path: str, session_id: str):
    """Generate TTS audio"""
    if tts_preference == "coqui":
        tts_url = "http://tts:8000/generate"
        tts_response = requests.post(tts_url,
                                     json={"text": text, "model": "tts_models/multilingual/multi-dataset/xtts_v2"})
        tts_response.raise_for_status()
        audio_path = f"{out_path}/{session_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_response.content)
        return audio_path
    else:
        from elevenlabs.client import ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("VOICE_ID")

        print(f"api_key: {api_key}")
        print(f"voice_id: {voice_id}")

        elevenlabs = ElevenLabs(api_key=api_key)
        response = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
        )

        print("Saving audio file...")
        audio_path = f"{out_path}/{session_id}.mp3"
        with open(audio_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"Audio file saved to {audio_path}")

        return audio_path


@app.post("/generate")
async def predict_image(
        background_tasks: BackgroundTasks,
        text: str = Form(...),
        user_id: str = Form(...),
        size: str = Form("256"),
        tts_preference: str = Form("elevenlabs"),
        stream: bool = Form(...)):

    out_path = f"/app/output/{user_id}"
    os.makedirs(out_path, exist_ok=True)

    temp_files = []

    # Create session
    session_id = uuid.uuid4()

    # Save image
    if user_id:
        img_path = f"/app/img/{user_id}.jpg"
        if not os.path.exists(img_path):
            # create new image from default avatar
            with open("/app/img/avatar.png", "rb") as f:
                with open(img_path, "wb") as f2:
                    f2.write(f.read())
    else:
        img_path = "/app/img/avatar.png"

    # Generate TTS
    audio_path = generate_tts(text, tts_preference, out_path, str(session_id))
    temp_files.append(audio_path)

    process = await asyncio.create_subprocess_exec(
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", img_path,
        "--result_dir", f"{out_path}/{session_id}",
        "--preprocess", "full",
        "--facerender", "pirender",
        "--enhancer", "gfpgan",
        "--still",
    )
    await process.wait()

    # Schedule cleanup after the response is sent
    background_tasks.add_task(cleanup_files, temp_files)

    if stream:
        def video_stream():
            with open(out_path, "rb") as s:
                while chunk := s.read(8192):
                    yield chunk

        return StreamingResponse(video_stream(), media_type="video/mp4")

    return FileResponse(out_path, media_type="video/mp4", filename="result.mp4")


def populate_temp_files(temp_files: list, out_path, user_id, session_id: str):
    temp_files.append(f"{out_path}/coeff##{session_id}.wav")
    temp_files.append(f"{out_path}/{session_id}.wav")
    temp_files.append(f"{out_path}/coeff##{session_id}.txt")
    temp_files.append(f"{out_path}/coeff##{session_id}.mat")
    temp_files.append(f"{out_path}/coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_{user_id}##{session_id}.mp4")
    temp_files.append(f"{out_path}/{user_id}##{session_id}.txt")
    temp_files.append(f"{out_path}/{user_id}##{session_id}.mp4")
    temp_files.append(f"{out_path}/{user_id}##{session_id}.mat")


@app.post("/presave-photo")
async def upload_photo(user_id: str = Form(...), image: UploadFile = File(...)):
    os.makedirs("/app/img", exist_ok=True)
    img_path = f"/app/img/{user_id}.jpg"

    with open(img_path, "wb") as f:
        f.write(await image.read())

    return {"message": "Photo uploaded successfully", "user_id": user_id}


@app.get("/health")
async def health_check():
    try:
        print("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
