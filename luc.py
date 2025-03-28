import torch
import cv2
import os
import shutil
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageFile, UnidentifiedImageError
from io import BytesIO
from transformers import AutoModelForImageClassification, ViTImageProcessor
from typing import Dict, Any, List
import numpy as np

# Configure PIL to be more tolerant of image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load NSFW Model - Using a more robust model
MODEL_NAME = "Falconsai/nsfw_image_detection"
try:
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on device: {device}")
    
    # Verify model labels
    print("Model class labels:", model.config.id2label)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

def is_valid_image(file_bytes: bytes) -> bool:
    """Improved image validation"""
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img.verify()
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
        return True
    except (UnidentifiedImageError, Exception) as e:
        print(f"Invalid image: {str(e)}")
        return False

def is_valid_video(file_path: str) -> bool:
    """Improved video validation"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        
        # Check first frame
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception as e:
        print(f"Video validation error: {str(e)}")
        return False

def classify_image(image: Image.Image) -> Dict[str, Any]:
    """Enhanced image classification with debug info"""
    try:
        start_time = time.time()
        
        # Enhanced preprocessing
        inputs = processor(
            images=image, 
            return_tensors="pt",
            do_resize=True,
            do_normalize=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        predicted_idx = logits.argmax(-1).item()
        classification = model.config.id2label[predicted_idx]
        confidence = float(probabilities[0][predicted_idx].item())
        
        # Debug output
        print("\n=== Model Classification Debug ===")
        print(f"Input shape: {inputs['pixel_values'].shape}")
        print(f"Raw logits: {logits.cpu().numpy()}")
        print(f"Probabilities: {probabilities.cpu().numpy()}")
        print("Class probabilities:")
        for i, label in model.config.id2label.items():
            print(f"{label}: {probabilities[0][i].item():.2%}")
        
        # Confidence threshold adjustment
        if classification == "SFW" and probabilities[0][1] > 0.3:  # If NSFW prob > 30%
            print("Potential misclassification detected - adjusting confidence")
            classification = "NSFW"
            confidence = float(probabilities[0][1].item())
        
        return {
            "classification": classification,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "all_probabilities": {
                label: float(prob.item()) 
                for label, prob in zip(model.config.id2label.values(), probabilities[0])
            }
        }
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        raise RuntimeError(f"Image classification failed: {str(e)}")

def extract_frames(video_path: str, frame_rate: int = 2) -> List[Any]:
    """Improved frame extraction with error handling"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps) // frame_rate) if fps > 0 else 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_idx % frame_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append((frame_idx, pil_image))
                except Exception as e:
                    print(f"Frame {frame_idx} error: {str(e)}")
        
        cap.release()
        return frames
    except Exception as e:
        raise RuntimeError(f"Video processing failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/classify/")
async def classify_media(file: UploadFile = File(...)):
    """Endpoint for classifying media files"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Read file content
        file_content = await file.read()
        
        # First try to process as image
        if is_valid_image(file_content):
            try:
                image = Image.open(BytesIO(file_content)).convert("RGB")
                result = classify_image(image)
                
                # Force NSFW if probability is high, regardless of label
                if result["all_probabilities"].get("nsfw", 0) > 0.7:
                    result["classification"] = "NSFW"
                    result["confidence"] = result["all_probabilities"]["nsfw"]
                
                return {
                    "status": "success",
                    "filename": file.filename,
                    "file_type": "image",
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "processing_time": f"{result['processing_time']:.2f}s",
                    "details": result["all_probabilities"]
                }
            except Exception as e:
                # If image processing fails, try as video
                pass
        
        # Save to temp file for video processing
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Try to process as video
        if is_valid_video(temp_file_path):
            start_time = time.time()
            frames = extract_frames(temp_file_path, frame_rate=2)
            
            if not frames:
                raise HTTPException(400, detail="No frames extracted from video")
            
            classifications = []
            nsfw_count = 0
            total_confidence = 0.0
            max_nsfw_confidence = 0.0
            
            for idx, img in frames:
                result = classify_image(img)
                classifications.append({
                    "frame": idx,
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "processing_time": f"{result['processing_time']:.2f}s",
                    "probabilities": result["all_probabilities"]
                })
                
                # Count as NSFW if either:
                # 1. Classification is NSFW, or
                # 2. NSFW probability > 70% (even if classified as SFW)
                nsfw_prob = result["all_probabilities"].get("nsfw", 0)
                if result["classification"].lower() == "nsfw" or nsfw_prob > 0.7:
                    nsfw_count += 1
                    total_confidence += nsfw_prob
                    if nsfw_prob > max_nsfw_confidence:
                        max_nsfw_confidence = nsfw_prob
            
            # Final classification - if any frame is NSFW, whole video is NSFW
            final_classification = "NSFW" if nsfw_count > 0 else "SFW"
            avg_confidence = total_confidence / max(1, nsfw_count) if nsfw_count > 0 else 0
            
            return {
                "status": "success",
                "filename": file.filename,
                "file_type": "video",
                "frame_classifications": classifications,
                "final_classification": final_classification,
                "confidence": round(max_nsfw_confidence, 4) if final_classification == "NSFW" else None,
                "total_processing_time": f"{round(time.time() - start_time, 4)}s",
                "nsfw_frame_count": nsfw_count,
                "total_frames": len(frames),
                "details": {
                    "max_nsfw_confidence": max_nsfw_confidence,
                    "avg_nsfw_confidence": avg_confidence
                }
            }
        
        # If neither image nor video worked
        raise HTTPException(400, detail="Unsupported or corrupted file format")
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"Internal server error: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)