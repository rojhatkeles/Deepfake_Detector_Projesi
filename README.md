# 🕵️ Deepfake Detector Projesi (XAI Destekli)

Modern çağın siber güvenlik problemlerinden biri olan sentetik/üretilmiş yüzleri (Deepfake) tespit etmek için geliştirilmiş, **Açıklanabilir Yapay Zeka (XAI - Grad-CAM)** destekli derin öğrenme sistemidir.

## 🚀 Proje Başarı Özeti (State-of-the-Art)
Bu proje, devasa **140.000 fotoğraflı** Kaggle veri okyanusunda, Transfer Learning (VGG16) kullanılarak yerel ortamda Apple Silicon (MPS) hızlandırmasıyla başarıyla eğitilmiş ve otonom bir Streamlit arayüzüne dönüştürülmüştür.

*   **Eğitim Verisi:** 140.000 Görsel (Kaggle 'Real vs Fake Faces' Arşivi)
*   **Mimari:** VGG16 (Custom Sequential Classifier + %50 Dropout regularization)
*   **Arama / Kırpma Motoru:** Otonom Bypass Mekanizmalı Facenet-PyTorch (MTCNN) 
*   **Nihai Test Skoru (20.000 yepyeni görsel üzerinden):** **%98.94 Genel Doğruluk (Accuracy)**

---

## 📊 Karmaşıklık Matrisi (Confusion Matrix) Sonuçları
Model 20.000 test fotoğrafını taramış ve mükemmele yakın bir dağılımla, "gerçeğe sahte deme (Paranoya)" veya "sahteye kanma (Körlük)" ön yargülerine kapılmadan (Biased olmadan) ideal sınırları bulmuştur:

*   **True Negatives (Gerçeği bilen):** 9896 / 10000
*   **True Positives (Sahteyi avlayan):** 9891 / 10000
*   **False Positives (Paranoya):** 104
*   **False Negatives (Kandırılan):** 109

## 🖥️ Arayüz Sistemi (Otonom Kesim & XAI)
Kullanıcı sisteme bir resim yüklediğinde arkada 3 adımlı bir koruma duvarı çalışır:
1.  **Otonom Yüz Analizi:** MTCNN, fotoğrafın analizini yapar. Eğer yüz ekranın %80'inden fazlasını kaplıyorsa (Laboratuvar/Zoom formatlı veri), çift kırpmayı (Double-Crop) engellemek için kendini uyumlu hale getirir ve by-pass yapar. Çevreleyen bir alan varsa MTCNN agresif `%10 Margin` (Sıfıra sıfır) sistemiyle kulak/saç sınırını koparıp alır.
2.  **Karar (Inference) Ağı:** Kesilen tensor yüzeyine `torch.no_grad()` ile VGG16 uygulanır. Tespit eşiği, siber güvenlik hassasiyeti için `0.40`'a çekildi.
3.  **Grad-CAM (Isı Haritası):** Eğer Deepfake tespiti yapılırsa XAI devreye girer ve modelin o fotoğrafta hangi pürüzlü piksellere (Saç sınırları, asimetrik arka parıltılar vb.) kanıp o kararı verdiğini sıcak kırmızı piksellerle (Heatmap) kanıtlar.

## ⚙️ Kurulum & Kullanım
Python 3.12+ (Anaconda/Venv)

```bash
# Kütüphaneleri kurun
pip install -r requirements.txt
pip install grad-cam

# Arayüzü Başlatın (Kendi localhost'unuzda)
streamlit run ui/app.py
```
> Not: `saved_models/best_model.pth` (~520MB) Git LFS kısıtı nedeniyle bu repoda maskelenmiştir. Modelin lokalinizde eğitilmesi tavsiye edilir.
