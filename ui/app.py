import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
import sys

# src klasöründeki modülleri çağırabilmek için projeye sistem yolu entegre ediliyor
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src'))

from model import get_deepfake_detector_model
from explainer import get_gradcam_heatmap
from facenet_pytorch import MTCNN

# 1. Arayüz Ayarları (UI Configurations)
st.set_page_config(page_title="Deepfake Dedektörü XAI", page_icon="🕵️", layout="wide")

# Modern, karanlık ve neon kırmızı odaklı tasarım (#ff4b4b Streamlit kırmızısıdır)
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
    }
    .metric-card {
        background: rgba(30,30,30, 0.8);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: fadeIn 1s;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(#eee, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 50px;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# 2. Modellerin Ön Yüklenmesi (Bellekte Cache)
@st.cache_resource
def load_ai_models():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Türevlerin alınabilmesi için XAI modelinde freeze_features=False!
    model = get_deepfake_detector_model(freeze_features=False) 
    weights_path = os.path.join(project_root, 'saved_models', 'best_model.pth')
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
    model = model.to(device)
    model.eval()
    
    # Canlı Yüz Tespiti (MTCNN) Motoru
    # margin=40: Kulakları, çene hattını ve saç sınırlarını kaybetmemek için İdeal Altın Oran.
    mtcnn_model = MTCNN(image_size=224, margin=40, keep_all=False, post_process=False, device=torch.device('cpu'))
    
    return model, mtcnn_model, device

model, mtcnn, device = load_ai_models()

# 3. Yükleme Ekranı
st.title("Derin Öğrenme Tabanlı Deepfake Dedektörü")
st.markdown("<p style='text-align:center;'>🔍 Apple Silicon M2 Pro & PyTorch ile geliştirilmiştir. (Açıklanabilir Yapay Zeka Desteği)</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Test etmek için bir yüz fotoğrafı yükleyin (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert('RGB')
    
    # 3 Kolonlu Dashboard Tasarımı
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4 style='text-align:center; color:#ccc'>1. Orijinal Girdi</h4>", unsafe_allow_html=True)
        st.image(original_img, use_column_width=True)
        
    # Otonom Kırpma Zekası (Auto-Bypass)
    face_img = None
    with st.spinner("AI Yüz Geometrisini Hesaplıyor..."):
        boxes, probs = mtcnn.detect(original_img)
        
        if boxes is not None and len(boxes) > 0:
            # En büyük/ilk yüzün ekranda kapladığı alanı hesapla
            box = boxes[0]
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            img_w, img_h = original_img.size
            
            # Eğer yüz fotoğrafın %80'inden fazlasını kaplıyorsa (Yani zaten KOCAMAN bir yüzse makası iptal et)
            if (box_w / img_w) > 0.80 or (box_h / img_h) > 0.80:
                # Model Çift-Kırpma cehennemine düşmesin diye makası otonom iptal et:
                face_img = original_img.resize((224, 224))
            else:
                # Arka plan az bile olsa acıma, MTCNN ile sıkıca kırp (Tight Crop)!
                face_tensor = mtcnn(original_img)
                if face_tensor is not None:
                    face_img = Image.fromarray(face_tensor.permute(1, 2, 0).byte().numpy())
        
    if face_img is None:
        st.error("❌ Sistemin MTCNN motoru bu fotoğrafta yüz tespit edemedi! Kamera uzaksa lütfen 'Zaten kırpılmış' kutucuğunu işaretleyin.")
    else:
        
        with col2:
            st.markdown("<h4 style='text-align:center; color:#ccc'>2. İzole Sistem (Crop)</h4>", unsafe_allow_html=True)
            st.image(face_img, use_column_width=True)
            
        # Makine Öğrenimi (AI) Karar Mekanizması
        with st.spinner("AI Derin Analiz Yapıyor (Hassas Mod)..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(face_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                
            # AGRESİF TESPİT: 0.40 ve üzeri FAKE olarak değerlendirilir!
            is_fake = prob > 0.40
            confidence = prob * 100 if is_fake else (1 - prob) * 100
            
            # XAI Isı Haritası (Grad-CAM)
            heatmap_pil = get_gradcam_heatmap(model, input_tensor, face_img)
            
            with col3:
                st.markdown("<h4 style='text-align:center; color:#ccc'>3. Grad-CAM Haritası</h4>", unsafe_allow_html=True)
                st.image(heatmap_pil, use_column_width=True)
                
            st.markdown("---")
            # Sonuç Kutuları (Dynamic UX)
            if is_fake:
                st.markdown(f"<div class='metric-card' style='border: 2px solid #ff4b4b; background: rgba(255, 75, 75, 0.1)'><h2>🚨 DİKKAT: %{confidence:.2f} DEEPFAKE TESPİT EDİLDİ</h2><p>Model matris özelliklerini inceledi ve yukarıdaki zihin haritasındaki renkli (kızıl/sarı) noktalardaki doku anormalliklerine odaklanarak sahtelik kararı verdi.</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-card' style='border: 2px solid #00f260; background: rgba(0, 242, 96, 0.1)'><h2>✅ GÜVENİLİR: %{confidence:.2f} ORANINDA GERÇEK YÜZ</h2><p>Ağ algılayıcıları resimde Deepfake pürüzlerine veya sentetik piksellere rastlamadı.</p></div>", unsafe_allow_html=True)
