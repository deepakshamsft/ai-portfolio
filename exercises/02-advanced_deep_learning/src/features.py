"""Image Preprocessing and Feature Engineering

Handles image preprocessing, normalization, and resizing operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
import torch
from src.utils import setup_logger


logger = setup_logger()


class ImagePreprocessor:
    """
    Image preprocessing pipeline for computer vision models.
    """
    
    def __init__(self, config: dict):
        """
        Initialize image preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.image_size = tuple(config['preprocessing']['image_size'])
        self.normalize_mean = config['preprocessing']['normalize_mean']
        self.normalize_std = config['preprocessing']['normalize_std']
        
        logger.info(f"Initialized preprocessor with size {self.image_size}")
    
    def preprocess(
        self,
        image: Union[np.ndarray, str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Input image (numpy array or file path)
            normalize: Whether to apply normalization
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = self.resize(image, self.image_size)
        
        # Normalize
        if normalize:
            image = self.normalize(image)
        
        # Convert to tensor and add batch dimension
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image
    
    def preprocess_batch(
        self,
        images: list,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images (numpy arrays or file paths)
            normalize: Whether to apply normalization
            
        Returns:
            Batch of preprocessed image tensors
        """
        processed = [self.preprocess(img, normalize) for img in images]
        return torch.stack(processed)
    
    @staticmethod
    def resize(
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (height, width)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean and std.
        
        Args:
            image: Input image (0-255 range)
            
        Returns:
            Normalized image
        """
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        mean = np.array(self.normalize_mean, dtype=np.float32)
        std = np.array(self.normalize_std, dtype=np.float32)
        
        image = (image - mean) / std
        
        return image
    
    @staticmethod
    def denormalize(
        image: Union[torch.Tensor, np.ndarray],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Denormalize image for visualization.
        
        Args:
            image: Normalized image tensor or array
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Denormalized image (0-255 range)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Handle batch dimension
        if image.ndim == 4:
            image = image[0]
        
        # Convert from CHW to HWC
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        
        # Denormalize
        image = (image * std) + mean
        
        # Scale to 0-255
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    def pad_to_square(image: np.ndarray, pad_value: int = 114) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Pad image to square shape (useful for YOLO models).
        
        Args:
            image: Input image
            pad_value: Padding value
            
        Returns:
            Tuple of (padded image, (pad_h, pad_w))
        """
        h, w = image.shape[:2]
        size = max(h, w)
        
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        
        padded = cv2.copyMakeBorder(
            image,
            pad_h, size - h - pad_h,
            pad_w, size - w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value)
        )
        
        return padded, (pad_h, pad_w)
    
    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with unchanged aspect ratio using padding (letterbox).
        
        Args:
            image: Input image
            new_shape: Target shape (height, width)
            color: Padding color
            
        Returns:
            Tuple of (resized image, scale ratio, (dw, dh))
        """
        shape = image.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        
        return image, r, (dw, dh)


def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1
) -> np.ndarray:
    """
    Apply color jittering for data augmentation.
    
    Args:
        image: Input image
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        hue: Hue jitter factor
        
    Returns:
        Jittered image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Apply random adjustments
    if brightness > 0:
        hsv[:, :, 2] *= np.random.uniform(1 - brightness, 1 + brightness)
    
    if saturation > 0:
        hsv[:, :, 1] *= np.random.uniform(1 - saturation, 1 + saturation)
    
    if hue > 0:
        hsv[:, :, 0] += np.random.uniform(-hue * 180, hue * 180)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return image
