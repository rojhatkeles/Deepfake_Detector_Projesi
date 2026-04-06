#!/bin/bash

# Tüm renk kodları ve UI hazırlıkları
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}🚀 LensAI Enterprise - Auto-Start Protocol 🚀${NC}"
echo -e "${BLUE}==============================================${NC}"

# Aşama 1: Ortamdaki tüm hayalet (eski) portları öldürerek temiz bir sayfa açıyoruz
echo -e "\n🧹 ${RED}Gereksiz portlar ve eski oturumlar temizleniyor...${NC}"
lsof -ti:8089 | xargs kill -9 2>/dev/null
lsof -ti:8090 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1
echo -e "✅ Temizlik tamamlandı!\n"

# Aşama 2: Python Backend (Yapay Zeka Motorunu Arka Planda Çalıştır)
echo -e "🧠 ${GREEN}[1/2] Yapay Zeka Derin Öğrenme API'si (FastAPI) Uyandırılıyor...${NC}"
cd /Users/rojhat/Desktop/Deepfake_Detector_Projesi
source .venv/bin/activate
uvicorn api:app --reload --port 8089 > /dev/null 2>&1 &
BACKEND_PID=$!
sleep 2 # Motorun ısınması için bekleme süresi

# Aşama 3: Node.js Frontend (Kullanıcı Arayüzü)
echo -e "🖥️  ${GREEN}[2/2] Reaktif Kullanıcı Arayüzü (React/Vite) Yükleniyor...${NC}"
cd frontend
npm run dev -- --port 8090 > /dev/null 2>&1 &
FRONTEND_PID=$!
sleep 2

# Başarı Ekranı
echo -e "\n${BLUE}========================================================================${NC}"
echo -e "${GREEN}🌟 SİSTEM KUSURSUZ ŞEKİLDE YAYINDA! (Tüm Motorlar Aktif) 🌟${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e "👉 ${GREEN}Lütfen Tıklayın ve Açın: http://localhost:8090${NC}"
echo -e "------------------------------------------------------------------------"
echo -e "⚠️  (Sistemi tamamen durdurmak/kapatmak için ${RED}CTRL + C${NC} yapınız)"

# Tüm arka plan süreçlerini düzgün kapatmak için kanca (Trap)
trap "echo -e '\n🛑 ${RED}LensAI Sistemi Tamamen Kapatılıyor... Lütfen bekleyin.${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Süreçleri açık tut
wait 
