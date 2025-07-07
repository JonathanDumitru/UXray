"""
FastAPI backend for UXray heatmap generation service.

Provides REST API endpoints for generating predictive heatmaps
using academic datasets and state-of-the-art saliency models.
"""

import os
import base64
import io
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image

# Import our custom modules
from models.deep_gaze import DeepGazeModel
from models.salicon_net import SaliconNetModel
from datasets.mit1003 import MIT1003Dataset
from datasets.salicon import SALICONDataset
from datasets.osie import OSIEDataset

# Initialize FastAPI app
app = FastAPI(
    title="UXray Heatmap Generator",
    description="Predictive UX heatmap generation using academic datasets",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models = {
    "deep_gaze": None,
    "salicon_net": None
}

# Dataset instances
datasets = {
    "mit1003": None,
    "salicon": None,
    "osie": None
}


@app.on_event("startup")
async def startup_event():
    """Initialize models and datasets on startup."""
    print("Initializing UXray Heatmap Generator...")
    
    # Initialize models
    try:
        models["deep_gaze"] = DeepGazeModel()
        models["salicon_net"] = SaliconNetModel()
        print("Models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")
    
    # Initialize datasets (if data is available)
    data_dir = os.getenv("DATASET_DIR", "./data")
    if os.path.exists(data_dir):
        try:
            datasets["mit1003"] = MIT1003Dataset(os.path.join(data_dir, "mit1003"))
            datasets["salicon"] = SALICONDataset(os.path.join(data_dir, "salicon"))
            datasets["osie"] = OSIEDataset(os.path.join(data_dir, "osie"))
            print("Datasets initialized successfully")
        except Exception as e:
            print(f"Error initializing datasets: {e}")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "UXray Heatmap Generator",
        "version": "1.0.0",
        "description": "Predictive UX heatmap generation using academic datasets",
        "endpoints": {
            "generate_heatmap": "/generate-heatmap",
            "models": "/models",
            "datasets": "/datasets",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {name: model.is_loaded if model else False for name, model in models.items()},
        "datasets_available": {name: dataset is not None for name, dataset in datasets.items()}
    }


@app.get("/models")
async def get_models():
    """Get information about available models."""
    model_info = {}
    for name, model in models.items():
        if model:
            model_info[name] = model.get_model_info()
        else:
            model_info[name] = {"status": "not_loaded"}
    
    return {
        "models": model_info,
        "total_models": len(models)
    }


@app.get("/datasets")
async def get_datasets():
    """Get information about available datasets."""
    dataset_info = {}
    for name, dataset in datasets.items():
        if dataset:
            dataset_info[name] = dataset.get_dataset_info()
        else:
            dataset_info[name] = {"status": "not_available"}
    
    return {
        "datasets": dataset_info,
        "total_datasets": len(datasets)
    }


@app.post("/generate-heatmap")
async def generate_heatmap(
    file: UploadFile = File(...),
    model_name: str = "deep_gaze",
    alpha: float = 0.7,
    return_overlay: bool = True
):
    """
    Generate predictive heatmap for uploaded image.
    
    Args:
        file: Uploaded image file
        model_name: Name of the model to use ('deep_gaze' or 'salicon_net')
        alpha: Transparency factor for heatmap overlay (0.0 to 1.0)
        return_overlay: Whether to return heatmap overlay or just saliency map
        
    Returns:
        JSON response with heatmap data and metadata
    """
    # Validate model name
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Available models: {list(models.keys())}"
        )
    
    # Get model
    model = models[model_name]
    if not model or not model.is_loaded:
        raise HTTPException(
            status_code=500,
            detail=f"Model {model_name} is not loaded"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Convert RGBA to RGB if necessary
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Generate heatmap
        if return_overlay:
            result_image = model.generate_heatmap(image_array, alpha)
        else:
            saliency_map = model.predict(image_array)
            result_image = model.postprocess_saliency(saliency_map, image_array.shape[:2])
        
        # Convert result to base64
        result_pil = Image.fromarray((result_image * 255).astype(np.uint8))
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Get model info
        model_info = model.get_model_info()
        
        return {
            "success": True,
            "model_used": model_name,
            "model_info": model_info,
            "image_size": image_array.shape,
            "heatmap_data": result_base64,
            "metadata": {
                "alpha": alpha,
                "return_overlay": return_overlay,
                "processing_time": "N/A"  # Could add timing in future
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating heatmap: {str(e)}"
        )


@app.post("/batch-generate")
async def batch_generate_heatmaps(
    files: list[UploadFile] = File(...),
    model_name: str = "deep_gaze",
    alpha: float = 0.7
):
    """
    Generate heatmaps for multiple images in batch.
    
    Args:
        files: List of uploaded image files
        model_name: Name of the model to use
        alpha: Transparency factor for heatmap overlay
        
    Returns:
        JSON response with heatmaps for all images
    """
    # Validate model
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Available models: {list(models.keys())}"
        )
    
    model = models[model_name]
    if not model or not model.is_loaded:
        raise HTTPException(
            status_code=500,
            detail=f"Model {model_name} is not loaded"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "File is not an image"
                })
                continue
            
            # Process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Convert RGBA to RGB if necessary
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Generate heatmap
            result_image = model.generate_heatmap(image_array, alpha)
            
            # Convert to base64
            result_pil = Image.fromarray((result_image * 255).astype(np.uint8))
            buffer = io.BytesIO()
            result_pil.save(buffer, format="PNG")
            result_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            results.append({
                "index": i,
                "filename": file.filename,
                "success": True,
                "heatmap_data": result_base64,
                "image_size": image_array.shape
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 