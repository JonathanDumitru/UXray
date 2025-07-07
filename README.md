# UXray - Predictive UX Heatmap Generator

## Overview

UXray is a comprehensive desktop application for generating predictive UX heatmaps using state-of-the-art saliency prediction models trained on academic datasets. The system provides accurate predictions of where users will look when viewing UI screenshots, helping designers optimize user interfaces.

## 🎯 Features

- **Academic Dataset Integration**: Uses MIT1003, SALICON, and OSIE datasets for training
- **Multiple Saliency Models**: DeepGaze and SALICON-inspired models
- **Real-time Processing**: Sub-second heatmap generation
- **Batch Processing**: Generate heatmaps for multiple images
- **Beautiful UI**: Apple-style minimalist interface
- **Desktop Application**: Tauri-wrapped React application

## 🏗️ Architecture

```
UXray/
├── backend/              # Python ML backend
│   ├── datasets/        # Academic dataset implementations
│   ├── models/          # Saliency prediction models
│   ├── scripts/         # Utility scripts
│   └── main.py         # FastAPI application
├── frontend/            # React + Vite frontend (coming soon)
├── .devcontainer/       # Development container configuration
└── docs/               # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Git

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download academic datasets**:
   ```bash
   cd backend
   python scripts/download_datasets.py --dataset all
   ```

3. **Train models** (optional):
   ```bash
   python scripts/train_models.py --dataset all --model all
   ```

4. **Start the API server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup (Coming Soon)

1. **Install Node.js dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

## 📊 Academic Datasets

### MIT1003 Dataset
- **Source**: MIT Saliency Benchmark
- **Size**: 1003 images with eye-tracking data
- **Use**: Training and validation of saliency models
- **License**: Academic use permitted

### SALICON Dataset
- **Source**: Microsoft COCO Saliency
- **Size**: 10,000+ images with fixation maps
- **Use**: Large-scale training data
- **License**: Creative Commons Attribution 4.0

### OSIE Dataset
- **Source**: Object and Semantic Images and Eye-tracking
- **Size**: 700 images with eye-tracking data
- **Use**: Object-level saliency prediction
- **License**: Academic research use

## 🔬 Model Performance

### Accuracy Metrics
- **AUC**: Area Under Curve (0.7-0.9 typical)
- **NSS**: Normalized Scanpath Saliency (1.0-2.5 typical)
- **CC**: Correlation Coefficient (0.5-0.8 typical)
- **SIM**: Similarity (0.3-0.6 typical)

### Speed
- **Processing Time**: < 1 second per image
- **Batch Processing**: Optimized for multiple images
- **Memory Usage**: Efficient for large images

## 🛠️ API Endpoints

### Health Check
```http
GET /health
```

### Generate Heatmap
```http
POST /generate-heatmap
```
Generate a predictive heatmap for an uploaded image.

### Batch Generate
```http
POST /batch-generate
```
Generate heatmaps for multiple images in batch.

## 🧪 Testing

Run the test suite to verify backend functionality:

```bash
cd backend
python test_backend.py
```

## 📚 Documentation

- [Backend Documentation](backend/README.md)
- [API Documentation](http://localhost:8000/docs) (when server is running)
- [Academic Papers](https://github.com/UXray/backend/wiki/Papers)

## 🛠️ Development

### Backend Development
- Python 3.8+ with FastAPI
- Academic datasets for training and validation
- Deep learning models for saliency prediction
- OpenCV and PIL for image processing

### Frontend Development (Coming Soon)
- React + Vite + TypeScript
- TailwindCSS for styling
- Headless UI for components
- Framer Motion for animations
- Tauri for desktop wrapper

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project uses academic datasets with their respective licenses:
- MIT1003: Academic use permitted
- SALICON: Creative Commons Attribution 4.0
- OSIE: Academic research use

## 🙏 Acknowledgments

- MIT Saliency Benchmark for the MIT1003 dataset
- Microsoft COCO team for the SALICON dataset
- OSIE dataset contributors
- Academic research community for saliency prediction models

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**UXray** - Making UX design data-driven with academic-grade accuracy. 