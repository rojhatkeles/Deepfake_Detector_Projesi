import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def get_gradcam_heatmap(model, input_tensor, original_image):
    """
    Modelin kararı hangi piksellere (ör: sahte cilt dokusu, gözdeki pürüz) bakarak verdiğini 
    açıklayan (Explainable AI - XAI) Grad-CAM üreteci.
    """
    
    # 1. Hedef Katmanı Belirlemek
    # VGG16 yapısında özellikleri (features) çıkaran mimarinin en uç ve derin katmanı sondaki Conv katmanıdır.
    # En yüksek seviyeli (high-level) özellikleri (göz, burun, ağız) burada toplar.
    target_layers = [model.features[-1]]
    
    # 2. Grad-CAM Motorunu Başlatmak
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. İleri Yayılım (Forward Pass) ve Türev Alma
    # Model girdiyi süzerken ağın sonunda çıkan kararın (Real/Fake),
    # target_layers katmanındaki nöronlara olan türevini (Gradient) hesaplar.
    # Türevi en yüksek olan pikseller, karara en çok etki eden piksellerdir.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # 4. Görüntüyü Maskelenebilir Formata Çevirme
    # Orijinal PIL Image nesnesini 224x224 RGB Float Matrisine Çevirmeliyiz (Değerler [0.0 - 1.0] arası olmalı)
    if isinstance(original_image, Image.Image):
        original_image = original_image.resize((224, 224))
        rgb_img = np.array(original_image, dtype=np.float32) / 255.0
    else:
        rgb_img = original_image
        
    # 5. Isı Haritası (Heatmap) Bindirmesi
    # Yüksek Türevli pikselleri KIRMIZI (Sıcak), Etkisiz pikselleri MAVİ (Soğuk) yapar.
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Streamlit (UI) tarafında daha kolay gösterebilmek için tekrar PIL Image Dönüşümü
    heatmap_pil = Image.fromarray(visualization)
    
    return heatmap_pil
