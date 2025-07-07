# UXray Backend - Generative Heatmap System

## Overview

The UXray backend provides predictive UX heatmap generation using state-of-the-art saliency prediction models trained on academic datasets. This system can analyze UI screenshots and generate heatmaps showing where users are likely to look, helping designers optimize user interfaces.

## Features

- **Academic Dataset Integration**: Uses MIT1003, SALICON, and OSIE datasets for training
- **Multiple Saliency Models**: DeepGaze and SALICON-inspired models
- **REST API**: FastAPI-based endpoints for heatmap generation
- **Batch Processing**: Generate heatmaps for multiple images
- **Real-time Processing**: Sub-second heatmap generation
- **Academic Accuracy**: Trained on university-grade datasets

## Architecture

```
backend/
├── datasets/           # Academic dataset implementations
│   ├── base.py        # Base dataset class
│   ├── mit1003.py     # MIT1003 dataset
│   ├── salicon.py     # SALICON dataset
│   └── osie.py        # OSIE dataset
├── models/            # Saliency prediction models
│   ├── saliency_model.py  # Base model class
│   ├── deep_gaze.py       # DeepGaze model
│   └── salicon_net.py     # SALICON model
├── scripts/           # Utility scripts
│   ├── download_datasets.py  # Dataset downloader
│   └── train_models.py      # Model training
├── main.py           # FastAPI application
└── requirements.txt  # Python dependencies
```

## Academic Datasets

### MIT1003 Dataset
- **Source**: MIT Saliency Benchmark
- **Size**: 1003 images with eye-tracking data
- **Use**: Training and validation of saliency models
- **License**: Academic use permitted
- **Reference**: Judd, T., et al. (2009). Learning to predict where humans look. ICCV.

### SALICON Dataset
- **Source**: Microsoft COCO Saliency
- **Size**: 10,000+ images with fixation maps
- **Use**: Large-scale training data
- **License**: Creative Commons Attribution 4.0
- **Reference**: Jiang, M., et al. (2015). SALICON: Saliency in Context. CVPR.

### OSIE Dataset
- **Source**: Object and Semantic Images and Eye-tracking
- **Size**: 700 images with eye-tracking data
- **Use**: Object-level saliency prediction
- **License**: Academic research use
- **Reference**: Xu, P., et al. (2014). Predicting human gaze beyond pixels. JAIR.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download academic datasets**:
   ```bash
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

## API Endpoints

### Health Check
```http
GET /health
```
Returns the health status of the service and loaded models.

### Get Models
```http
GET /models
```
Returns information about available saliency prediction models.

### Get Datasets
```http
GET /datasets
```
Returns information about available academic datasets.

### Generate Heatmap
```http
POST /generate-heatmap
```
Generate a predictive heatmap for an uploaded image.

**Parameters**:
- `file`: Image file (multipart/form-data)
- `model_name`: Model to use (`deep_gaze` or `salicon_net`)
- `alpha`: Transparency factor (0.0 to 1.0)
- `return_overlay`: Whether to return overlay or just saliency map

**Response**:
```json
{
  "success": true,
  "model_used": "deep_gaze",
  "model_info": {...},
  "image_size": [height, width, channels],
  "heatmap_data": "base64_encoded_image",
  "metadata": {
    "alpha": 0.7,
    "return_overlay": true,
    "processing_time": "N/A"
  }
}
```

### Batch Generate Heatmaps
```http
POST /batch-generate
```
Generate heatmaps for multiple images in batch.

**Parameters**:
- `files`: List of image files (multipart/form-data)
- `model_name`: Model to use
- `alpha`: Transparency factor

## Usage Examples

### Python Client
```python
import requests
import base64
from PIL import Image
import io

# Generate heatmap
url = "http://localhost:8000/generate-heatmap"
files = {"file": open("screenshot.png", "rb")}
data = {"model_name": "deep_gaze", "alpha": 0.7}

response = requests.post(url, files=files, data=data)
result = response.json()

if result["success"]:
    # Decode and save heatmap
    heatmap_data = base64.b64decode(result["heatmap_data"])
    heatmap_image = Image.open(io.BytesIO(heatmap_data))
    heatmap_image.save("heatmap.png")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/generate-heatmap" \
     -F "file=@screenshot.png" \
     -F "model_name=deep_gaze" \
     -F "alpha=0.7"
```

## Model Performance

### Accuracy Metrics
- **AUC**: Area Under Curve (0.7-0.9 typical)
- **NSS**: Normalized Scanpath Saliency (1.0-2.5 typical)
- **CC**: Correlation Coefficient (0.5-0.8 typical)
- **SIM**: Similarity (0.3-0.6 typical)

### Speed
- **Processing Time**: < 1 second per image
- **Batch Processing**: Optimized for multiple images
- **Memory Usage**: Efficient for large images

## Development

### Adding New Datasets
1. Create a new dataset class in `datasets/`
2. Inherit from `BaseSaliencyDataset`
3. Implement required methods
4. Add to dataset registry in `main.py`

### Adding New Models
1. Create a new model class in `models/`
2. Inherit from `SaliencyModel`
3. Implement `load_model()` and `predict()` methods
4. Add to model registry in `main.py`

### Training Custom Models
```bash
# Train on specific dataset
python scripts/train_models.py --dataset mit1003 --model deep_gaze

# Train all models on all datasets
python scripts/train_models.py --dataset all --model all

# Evaluate trained model
python scripts/train_models.py --evaluate --dataset mit1003 --model deep_gaze
```

## Configuration

### Environment Variables
- `DATASET_DIR`: Directory containing datasets (default: `./data`)
- `MODEL_DIR`: Directory containing trained models (default: `./models`)

### Model Configuration
Models can be configured in the training script:
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `epochs`: Number of training epochs
- `input_size`: Input image size for models

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Ensure datasets are downloaded
   - Check model files exist in models directory
   - Verify Python dependencies are installed

2. **Dataset errors**:
   - Run `python scripts/download_datasets.py --check`
   - Re-download datasets if needed
   - Verify dataset directory structure

3. **API errors**:
   - Check server logs for detailed error messages
   - Verify image format (JPEG, PNG supported)
   - Ensure file size is reasonable (< 10MB)

### Performance Optimization
- Use GPU acceleration for model inference
- Implement caching for frequently processed images
- Optimize image preprocessing pipeline
- Use batch processing for multiple images

## License

This project uses academic datasets with their respective licenses:
- MIT1003: Academic use permitted
- SALICON: Creative Commons Attribution 4.0
- OSIE: Academic research use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## Documentation

- [API Documentation](http://localhost:8000/docs) (when server is running)
- [Academic Papers](https://github.com/UXray/backend/wiki/Papers)
- [Model Architecture](https://github.com/UXray/backend/wiki/Models) 