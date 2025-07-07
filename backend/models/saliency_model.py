"""
Base saliency prediction model for generative heatmap generation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import cv2


class SaliencyModel(ABC):
    """
    Base class for saliency prediction models.
    
    Provides common functionality for preprocessing images,
    generating saliency maps, and post-processing results.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize saliency model.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load pre-trained model from path."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict saliency map for input image.
        
        Args:
            image: Input image array (H, W, 3) in RGB format
            
        Returns:
            Saliency map array (H, W) with values in [0, 1]
        """
        pass
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image array
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image array
        """
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            # Convert BGR to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size[::-1])  # cv2 uses (width, height)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def postprocess_saliency(self, saliency_map: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Post-process saliency map.
        
        Args:
            saliency_map: Raw saliency map from model
            original_size: Original image size (height, width)
            
        Returns:
            Post-processed saliency map
        """
        # Resize to original image size
        saliency_map = cv2.resize(saliency_map, original_size[::-1])
        
        # Normalize to [0, 1]
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        
        # Apply Gaussian smoothing for better visualization
        saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)
        
        return saliency_map
    
    def generate_heatmap(self, image: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """
        Generate heatmap overlay on original image.
        
        Args:
            image: Original image array
            alpha: Transparency factor for overlay
            
        Returns:
            Heatmap overlay image
        """
        # Predict saliency map
        saliency_map = self.predict(image)
        
        # Resize saliency map to match image size
        original_size = (image.shape[0], image.shape[1])
        saliency_map = self.postprocess_saliency(saliency_map, original_size)
        
        # Create heatmap colormap
        heatmap = cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.__class__.__name__,
            "is_loaded": self.is_loaded,
            "model_path": self.model_path
        } 