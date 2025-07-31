"""
Re-identification utilities for SMILEtrack.

This module provides utilities for extracting appearance features
and computing similarity for multi-object tracking with re-identification.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
import torch
import torchvision.transforms as transforms
from pathlib import Path


class SimpleReIDExtractor:
    """
    A simple re-identification feature extractor using basic image processing.
    
    For production use, this should be replaced with a trained deep learning model
    like OSNet, FastReID, or similar re-ID networks.
    """
    
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract re-ID features from a cropped detection.
        
        Args:
            image: Full frame image (H, W, C)
            bbox: Bounding box in (xmin, ymin, xmax, ymax) format
            
        Returns:
            Feature vector of size feature_dim
        """
        # Crop the detection from the image
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop the detection
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            # Return zero features for invalid crops
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Simple feature extraction using color and texture
        features = self._extract_simple_features(crop)
        
        return features
    
    def _extract_simple_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract simple features from a cropped detection.
        
        This is a placeholder implementation. In practice, you would use
        a trained re-ID model here.
        """
        # Resize to standard size
        crop_resized = cv2.resize(crop, (64, 128))
        
        # Color histogram features
        hist_features = []
        for channel in range(3):
            hist = cv2.calcHist([crop_resized], [channel], None, [16], [0, 256])
            hist_features.extend(hist.flatten())
        
        # Texture features using LBP-like patterns
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        texture_features = self._compute_texture_features(gray)
        
        # Combine all features
        all_features = np.concatenate([hist_features, texture_features])
        
        # Normalize and resize to target dimension
        if len(all_features) > self.feature_dim:
            # Downsample if too many features
            indices = np.linspace(0, len(all_features)-1, self.feature_dim, dtype=int)
            features = all_features[indices]
        else:
            # Pad if too few features
            features = np.pad(all_features, (0, self.feature_dim - len(all_features)), 'constant')
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features.astype(np.float32)
    
    def _compute_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute simple texture features."""
        # Simple gradient-based features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction histograms
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Histogram of gradients
        mag_hist, _ = np.histogram(magnitude, bins=8, range=(0, 255))
        dir_hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
        
        return np.concatenate([mag_hist, dir_hist]).astype(np.float32)


class DeepReIDExtractor:
    """
    Deep learning based re-ID feature extractor.
    
    This is a placeholder for integrating with trained re-ID models.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if path is provided
        if model_path and Path(model_path).exists():
            self._load_model()
    
    def _load_model(self):
        """Load the re-ID model from file."""
        # Placeholder for model loading
        # In practice, this would load a trained PyTorch model
        pass
    
    def extract_features(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract deep re-ID features from a detection.
        
        Args:
            image: Full frame image
            bbox: Bounding box coordinates
            
        Returns:
            Deep feature vector
        """
        if self.model is None:
            # Fallback to simple features if no model is loaded
            simple_extractor = SimpleReIDExtractor()
            return simple_extractor.extract_features(image, bbox)
        
        # Crop detection
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)  # Default feature dimension
        
        # Preprocess
        crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        # Extract features using the model
        with torch.no_grad():
            features = self.model(crop_tensor)
            features = features.cpu().numpy().flatten()
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features.astype(np.float32)


def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    if feat1 is None or feat2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(feat1, feat2) / (norm1 * norm2)


def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute Euclidean distance between two feature vectors."""
    if feat1 is None or feat2 is None:
        return float('inf')
    
    return np.linalg.norm(feat1 - feat2)


def compute_similarity_matrix(features1: List[np.ndarray], features2: List[np.ndarray]) -> np.ndarray:
    """
    Compute similarity matrix between two sets of features.
    
    Args:
        features1: List of feature vectors
        features2: List of feature vectors
        
    Returns:
        Similarity matrix of shape (len(features1), len(features2))
    """
    if not features1 or not features2:
        return np.zeros((len(features1), len(features2)))
    
    similarity_matrix = np.zeros((len(features1), len(features2)))
    
    for i, feat1 in enumerate(features1):
        for j, feat2 in enumerate(features2):
            similarity_matrix[i, j] = cosine_similarity(feat1, feat2)
    
    return similarity_matrix


def smooth_features(current_feat: np.ndarray, previous_feat: Optional[np.ndarray], alpha: float = 0.9) -> np.ndarray:
    """
    Smooth features using exponential moving average.
    
    Args:
        current_feat: Current frame features
        previous_feat: Previous smoothed features
        alpha: Smoothing factor
        
    Returns:
        Smoothed features
    """
    if previous_feat is None:
        return current_feat
    
    return alpha * previous_feat + (1 - alpha) * current_feat