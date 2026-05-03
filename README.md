\# 🔐 Image Encryption Project



\## 📌 Overview

This project implements image encryption using:

\- Arnold Cat Map (ACM)

\- SVD (Singular Value Decomposition)

\- Chaotic Sine Map

\- XOR Operation

\- Diffusion Layer



\## 🔁 Pipeline

Original → ACM → SVD → Chaos XOR → Diffusion → Encrypted



\## 🔓 Decryption

Reverse operations applied to recover original image.



\## 📊 Metrics Used

\- Entropy

\- Chi-Square Test



\## ▶️ How to Run



1\. Install dependencies:

pip install -r requirements.txt



2\. Run:

python main.py



## 📊 Results

### Encrypted + Decrypted
![Enc-Dec](outputs/ori-enc-dec.png)

### Histogram
![Histogram](outputs/histogram.png)

### Chi-Square
![Chi](outputs/chi-square.png)

### NPCR & UACI
![NPCR](outputs/NPCR&UACI.png)

### SSIM
![SSIM](outputs/SSIM1.png)
![SSIM](outputs/SSIM2.png)

### Correlation
![Correlation](outputs/correlation.png)

