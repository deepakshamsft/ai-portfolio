"""
Multimodal feature extraction
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image
from .models.clip import CLIPModel
from .models.whisper import WhisperModel
from .utils import setup_logger


logger = setup_logger(__name__)


class MultimodalFeatureExtractor:
    """
    Extract and fuse features from multiple modalities.
    """
    
    def __init__(
        self,
        clip_model: Optional[CLIPModel] = None,
        whisper_model: Optional[WhisperModel] = None,
        fusion_method: str = "concatenate"
    ):
        """
        Initialize feature extractor.
        
        Args:
            clip_model: CLIP model for text/image features
            whisper_model: Whisper model for audio features
            fusion_method: Method to fuse features ("concatenate", "average", "attention")
        """
        self.clip_model = clip_model
        self.whisper_model = whisper_model
        self.fusion_method = fusion_method
        
        logger.info(f"Initialized multimodal feature extractor with fusion: {fusion_method}")
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract text features using CLIP.
        
        Args:
            text: Input text
        
        Returns:
            Text feature vector
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not initialized")
        
        return self.clip_model.get_text_embedding(text)
    
    def extract_image_features(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Extract image features using CLIP.
        
        Args:
            image: Image path or PIL Image
        
        Returns:
            Image feature vector
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not initialized")
        
        return self.clip_model.get_image_embedding(image)
    
    def extract_audio_features(self, audio: Union[str, np.ndarray]) -> np.ndarray:
        """
        Extract audio features using Whisper encoder.
        
        Args:
            audio: Audio path or waveform
        
        Returns:
            Audio feature vector
        """
        if self.whisper_model is None:
            raise ValueError("Whisper model not initialized")
        
        # Transcribe and use text embedding as audio features
        # (Whisper encoder features could also be extracted directly)
        result = self.whisper_model.transcribe(audio)
        text = result["text"]
        
        return self.extract_text_features(text)
    
    def extract_multimodal_features(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        audio: Optional[Union[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from all available modalities.
        
        Args:
            text: Input text
            image: Input image
            audio: Input audio
        
        Returns:
            Dictionary mapping modality names to feature vectors
        """
        features = {}
        
        if text is not None:
            features["text"] = self.extract_text_features(text)
        
        if image is not None:
            features["image"] = self.extract_image_features(image)
        
        if audio is not None:
            features["audio"] = self.extract_audio_features(audio)
        
        logger.info(f"Extracted features from {len(features)} modalities")
        
        return features
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features from multiple modalities.
        
        Args:
            features: Dictionary mapping modality names to feature vectors
        
        Returns:
            Fused feature vector
        """
        if not features:
            raise ValueError("No features to fuse")
        
        feature_arrays = list(features.values())
        
        if self.fusion_method == "concatenate":
            # Simple concatenation
            fused = np.concatenate(feature_arrays, axis=0)
        
        elif self.fusion_method == "average":
            # Average pooling
            # Pad features to same length if needed
            max_len = max(f.shape[0] for f in feature_arrays)
            padded = []
            for f in feature_arrays:
                if f.shape[0] < max_len:
                    f = np.pad(f, (0, max_len - f.shape[0]))
                padded.append(f)
            
            fused = np.mean(padded, axis=0)
        
        elif self.fusion_method == "attention":
            # Learned attention weights (placeholder - would need training)
            # For now, use equal weights
            max_len = max(f.shape[0] for f in feature_arrays)
            padded = []
            for f in feature_arrays:
                if f.shape[0] < max_len:
                    f = np.pad(f, (0, max_len - f.shape[0]))
                padded.append(f)
            
            weights = np.ones(len(padded)) / len(padded)
            fused = np.average(padded, axis=0, weights=weights)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        logger.info(f"Fused {len(features)} features into shape {fused.shape}")
        
        return fused
    
    def compute_multimodal_similarity(
        self,
        query: Dict[str, Union[str, Image.Image, np.ndarray]],
        target: Dict[str, Union[str, Image.Image, np.ndarray]]
    ) -> float:
        """
        Compute similarity between two multimodal inputs.
        
        Args:
            query: Query modalities
            target: Target modalities
        
        Returns:
            Similarity score
        """
        # Extract features
        query_features = self.extract_multimodal_features(
            text=query.get("text"),
            image=query.get("image"),
            audio=query.get("audio")
        )
        
        target_features = self.extract_multimodal_features(
            text=target.get("text"),
            image=target.get("image"),
            audio=target.get("audio")
        )
        
        # Fuse features
        query_fused = self.fuse_features(query_features)
        target_fused = self.fuse_features(target_features)
        
        # Compute cosine similarity
        query_norm = query_fused / np.linalg.norm(query_fused)
        target_norm = target_fused / np.linalg.norm(target_fused)
        
        similarity = float(np.dot(query_norm, target_norm))
        
        return similarity
    
    def search_multimodal(
        self,
        query: Dict[str, Union[str, Image.Image, np.ndarray]],
        database: List[Dict[str, Union[str, Image.Image, np.ndarray]]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Search database with multimodal query.
        
        Args:
            query: Query modalities
            database: List of database items
            top_k: Number of results to return
        
        Returns:
            List of (index, similarity) tuples
        """
        # Extract query features
        query_features = self.extract_multimodal_features(
            text=query.get("text"),
            image=query.get("image"),
            audio=query.get("audio")
        )
        query_fused = self.fuse_features(query_features)
        query_norm = query_fused / np.linalg.norm(query_fused)
        
        # Compute similarities
        results = []
        for idx, item in enumerate(database):
            item_features = self.extract_multimodal_features(
                text=item.get("text"),
                image=item.get("image"),
                audio=item.get("audio")
            )
            item_fused = self.fuse_features(item_features)
            item_norm = item_fused / np.linalg.norm(item_fused)
            
            similarity = float(np.dot(query_norm, item_norm))
            results.append((idx, similarity))
        
        # Sort and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
