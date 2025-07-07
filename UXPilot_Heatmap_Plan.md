# UXPilot.ai Heatmap Generator — Project Plan

## Overview
A predictive UX heatmap generator leveraging university-grade datasets for high accuracy. The application will feature a modern, Apple-style desktop UI, and use state-of-the-art machine learning models for saliency prediction.

---

## Project Goals
- Generate predictive UX heatmaps for web/app UIs.
- Use academic/university-grade datasets to train/validate models.
- Provide a beautiful, Apple-style, responsive UI for uploading images/screens, viewing heatmaps, and exporting results.

---

## Core Features
- **Image Upload:** Users upload screenshots or UI images.
- **Heatmap Generation:** Predictive heatmap overlay using ML model.
- **Dataset Management:** Store, update, and reference academic datasets.
- **Result Export:** Download heatmaps as images or data.
- **User Auth:** (Optional) User accounts for saving history.
- **API:** Backend endpoint for heatmap generation.

---

## Tech Stack
- **Frontend:** React + Vite + TailwindCSS + Headless UI + Framer Motion
- **Backend:** Supabase (Auth, DB, Storage) + Python (ML/AI API, e.g., FastAPI)
- **State:** Zustand
- **Desktop Wrapper:** Tauri (Rust)
- **ML/AI:** Python (PyTorch/TensorFlow/Scikit-learn)
- **Datasets:** University-grade (e.g., MIT Saliency Benchmark, OSIE, SALICON, etc.)

---

## Architecture Overview
- **Frontend (Tauri/React):** UI for upload, display, and export.
- **Backend (Supabase):** User data, image storage, history.
- **ML Service (Python):** Receives images, runs predictive model, returns heatmap.
- **Dataset Storage:** Academic datasets for model training/validation.

---

## Task List

1. **Research and acquire top academic saliency/eye-tracking datasets** (e.g., MIT1003, SALICON, OSIE) and review their licenses for use.
2. **Select and implement a baseline saliency prediction model** (e.g., DeepGaze, SaliconNet, MLNet) and prepare for training.
3. **Train and validate the chosen model** using the acquired datasets, ensuring predictive accuracy.
4. **Export the trained model for inference** (ONNX, TorchScript, etc.) and set up a Python API (FastAPI/Flask) for heatmap generation.
5. **Set up the React + Vite + TailwindCSS frontend with Tauri wrapper**, including UI for image upload, heatmap display, and export functionality.
6. **Integrate the frontend with the backend API** to enable heatmap generation and display.
7. **(Optional) Implement Supabase Auth** for user accounts and history saving.
8. **Document every step:** dataset sources, model details, API, UI/UX, and deployment instructions.

---

## Documentation
- Maintain detailed documentation for setup, dataset sources, model details, API, UI/UX, and deployment. 