"""This module implements a FastAPI route for SAM2 based on point prompts specified as a list of coordinates. This can be used in conjunction with
AnomalyDINO for training-free defect segmentation. It returns a list of binary masks containing defects (class-agnostic).
"""
import os, json, numpy as np, torch, io, base64
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = build_sam2(os.getenv('CFG'), os.getenv('CKPT'), device=os.getenv('DEVICE'))
    predictor = SAM2ImagePredictor(model)
    app.state.predictor = predictor
    yield
    predictor = None

app = FastAPI(title="SAM2 API", lifespan=lifespan)

@app.post("/segment")
async def segment(
    image: UploadFile = File(..., description="RGB"),
    point_prompt: str = Form(..., description="JSON list [[x,y],...]"),
):
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(500, "Predictor not initialized")

    try:
        img = np.array(Image.open(image.file).convert("RGB"))
    except Exception as e:
        raise HTTPException(400, f"image couldn't be read: {e}")

    try:
        pts = np.array(json.loads(point_prompt), dtype=np.float32)
    except Exception:
        raise HTTPException(422, "point_prompt must be JSON list [[x,y],...]")

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HTTPException(422, "point_prompt must be list of [x,y]")

    predictor.set_image(img)
    labels = np.ones(len(pts), dtype=np.int64)
    mask_b64_list = []
    
    for i, p in enumerate(pts):
        masks, _, _ = predictor.predict(
            point_coords=p[None, :],
            point_labels=np.array([labels[i]]),
            box=None, mask_input=None, multimask_output=False,
        )

        mask = (masks[0] * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask)
        buffer = io.BytesIO()
        pil_mask.save(buffer, format="PNG")
        mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mask_b64_list.append(mask_b64)

    return {"count": len(mask_b64_list), "masks": mask_b64_list}
