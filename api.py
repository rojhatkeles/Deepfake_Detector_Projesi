import os
import torch
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from src.model import get_deepfake_detector_model
from src.explainer import get_gradcam_heatmap
from facenet_pytorch import MTCNN

# Gerçek bir SaaS B2B AI Uç Noktası (Endpoint) Seti
app = FastAPI(title="Deepfake API Core", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Vite React Frontend'in erişmesi için açık
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Engine Hazırlığı (Soğuk Başlangıç Optimizasyonu)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = get_deepfake_detector_model(freeze_features=False)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'best_model.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Otonom Makas (MTCNN)
mtcnn = MTCNN(image_size=224, margin=40, keep_all=False, post_process=False, device=torch.device('cpu'))

@app.get("/")
async def health_check():
    return {"status": "success", "message": "Enterprise Deepfake API is running perfectly. Awaiting POST streams at /analyze endpoint."}

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    
    contents = await file.read()
    original_img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 1. Yüz Tespiti (Otonom Double-Crop Koruması)
    boxes, probs = mtcnn.detect(original_img)
    face_img = None
    
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        img_w, img_h = original_img.size
        
        # Eğer yüz tüm ekranı kaplıyorsa bypass et, değilse marjinal tırpalama yap
        if (box_w / img_w) > 0.80 or (box_h / img_h) > 0.80:
            face_img = original_img.resize((224, 224))
        else:
            face_tensor = mtcnn(original_img)
            if face_tensor is not None:
                face_img = Image.fromarray(face_tensor.permute(1, 2, 0).byte().numpy())
    
    if face_img is None:
        raise HTTPException(status_code=400, detail="No biological human face detected in the image.")
        
    # 2. Makine Öğrenimi (Deep Analysis)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(face_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        
    is_fake = prob > 0.40
    confidence = prob * 100 if is_fake else (1 - prob) * 100
    
    # 3. Yorumlanabilir Yapay Zeka (XAI / Grad-CAM)
    heatmap_pil = get_gradcam_heatmap(model, input_tensor, face_img)
    
    # Frontend'e göndermek için Buffer kodlaması
    buffered = io.BytesIO()
    heatmap_pil.save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    face_buf = io.BytesIO()
    face_img.save(face_buf, format="JPEG")
    face_base64 = base64.b64encode(face_buf.getvalue()).decode("utf-8")
    
    return {
        "status": "success",
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "heatmap": f"data:image/jpeg;base64,{heatmap_base64}",
        "face_crop": f"data:image/jpeg;base64,{face_base64}"
    }
