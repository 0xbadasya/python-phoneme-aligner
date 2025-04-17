# 🔠 Phoneme Aligner for Ukrainian

A custom system for audio digitization and phoneme alignment by timestamps. Perfect as a personal development project with Python, audio processing, and ML.

---

## 📦 Installation

```bash
git clone https://github.com/yourname/phoneme-aligner.git
cd phoneme-aligner
pip install -r requirements.txt
🔧 FFmpeg is also required for mp3: https://ffmpeg.org/
```
---

## 🚀 Usage

```bash
python phoneme_aligner.py --audio audio/privit.mp3 --word привіт
```

or with customs phonemes:

```bash
python phoneme_aligner.py --audio audio/privit.mp3 --phonemes p,rʲ,i,vʲ,i,t
```
---

## 📂 Project Structure

audio/ — sample audio files (mp3, wav)

data/ — phoneme dictionaries

phoneme_aligner.py — main script

utils.py — helper functions

---

## 📌 Under the hood

- MFCC extraction using librosa
- Alignment algorithm — fastdtw
- Simple Ukrainian word dictionary → phoneme dictionary

**🎓 Created as a personal dtudying project.**