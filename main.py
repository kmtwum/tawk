import asyncio

from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import requests

import os
import torch
import gc
import uuid

from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract

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


def generate_tts(text: str, tts_preference: str, tts_voice_id: str, out_path: str, session_id: str):
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
        api_key = get_secret_key("ELEVENLABS_API_KEY_FILE")
        voice_id = os.getenv("VOICE_ID")
        if tts_voice_id:
            voice_id = tts_voice_id

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
        tts_preference: str = Form("elevenlabs"),
        tts_voice_id: str = Form(...),
        stream: bool = Form(True)):
    out_path = f"/app/output/{user_id}"
    os.makedirs(out_path, exist_ok=True)

    temp_files = []

    # Create session
    session_id = uuid.uuid4()

    # Save image
    if user_id:
        img_path = f"/app/user_img/{user_id}.jpg"
        if not os.path.exists(img_path):
            # download image from url
            print("Downloading image...")
            try:
                gcp_base = get_secret_key("GCP_BASE_FILE")
                response = requests.get(f"{gcp_base}/{user_id}")
                response.raise_for_status()
                print("Image downloaded successfully!")
                with open(img_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading image: {e}")
                img_path = "/app/img/avatar.jpg"
    else:
        img_path = "/app/img/avatar.jpg"

    # Generate TTS
    print("Generating TTS...")
    audio_path = generate_tts(text, tts_preference, tts_voice_id, out_path, str(session_id))
    temp_files.append(audio_path)

    print("Generating video...")
    video_path = process_video(out_path, img_path, audio_path, session_id)

    print(f"Video generated at {video_path}!")

    populate_temp_files(temp_files, out_path, user_id, str(session_id))

    # Schedule cleanup after the response is sent
    background_tasks.add_task(cleanup_files, temp_files)

    if stream:
        print("Streaming video...")
        file_size = os.path.getsize(video_path)

        def video_stream():
            with open(video_path, "rb") as s:
                while chunk := s.read(8192):
                    yield chunk

        return StreamingResponse(
            video_stream(),
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": "inline; filename=result.mp4"
            }
        )

    print("Sending video...")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="result.mp4",
        headers={"Accept-Ranges": "bytes"}
    )

def get_secret_key(secret):
    key_file = os.getenv(secret)
    if key_file:
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
            return api_key
    return None


def process_video(user_path, pic_path, audio_path, session_id):
    pre_process = "full"
    check_points = "/app/checkpoints"
    config_path = "/app/src/config"
    device = "cuda"
    renderer = "pirender"

    sad_talker_paths = init_path(check_points, config_path, "256", False, pre_process)
    preprocess_model = CropAndExtract(sad_talker_paths, device)

    audio_to_coeff = Audio2Coeff(sad_talker_paths, device)
    animate_from_coeff = AnimateFromCoeff_PIRender(sad_talker_paths, device)
    ref_eyeblink_coeff_path = None
    ref_pose_coeff_path = None

    meta_dir = os.path.join(user_path, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, meta_dir, pre_process, 256)

    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=True)
    coeff_path = audio_to_coeff.generate(batch, user_path, 1, ref_pose_coeff_path)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               32, None, None, None,
                               expression_scale=2, still_mode=True,
                               preprocess=pre_process, face_model=renderer, session=str(session_id))

    video_path = animate_from_coeff.generate(data, user_path, pic_path, crop_info,
                                             enhancer=None, background_enhancer=None,
                                             preprocess=pre_process, skip_background_blend=True)
    return video_path


def populate_temp_files(temp_files: list, out_path, user_id, session_id: str):
    temp_files.append(f"{out_path}/coeff##{session_id}.wav")
    temp_files.append(f"{out_path}/{session_id}.wav")
    temp_files.append(f"{out_path}/coeff##{session_id}.txt")
    temp_files.append(f"{out_path}/coeff##{session_id}.mat")
    temp_files.append(f"{out_path}/coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_coeff##{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_{user_id}##{session_id}.mp4")
    temp_files.append(f"{out_path}/{session_id}.mp4")
    temp_files.append(f"{out_path}/temp_{session_id}.mp4")


@app.post("/presave-photo")
async def upload_photo(user_id: str = Form(...), image: UploadFile = File(...)):
    os.makedirs("/app/user_img", exist_ok=True)
    img_path = f"/app/user_img/{user_id}.jpg"

    with open(img_path, "wb") as f:
        f.write(await image.read())

    return {"message": "Photo uploaded successfully", "user_id": user_id}


@app.get("/health")
async def health_check():
    try:
        print("health 200")
        return status.HTTP_200_OK

    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)