import asyncio
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from deepface import DeepFace

from app.core.config import get_settings
from app.core.exceptions import (
    FaceNotDetectedException,
    MultipleFacesException,
    InvalidImageException,
)
from app.utils.image_utils import bytes_to_numpy

settings = get_settings()

# Thread pool for CPU-bound face operations
_executor = ThreadPoolExecutor(max_workers=4)


class FaceService:
    """Service for face detection, embedding generation, and matching."""

    def __init__(self):
        self.model_name = settings.FACE_MODEL
        self.detector_backend = settings.FACE_DETECTOR
        self.threshold = settings.FACE_MATCH_THRESHOLD
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Ensure the face recognition model is loaded."""
        if not self._model_loaded:
            # Trigger model download/load by running a dummy embedding
            # This happens on first use
            try:
                dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
                DeepFace.represent(
                    dummy_img,
                    model_name=self.model_name,
                    detector_backend="skip",  # Skip detection for dummy
                    enforce_detection=False,
                )
                self._model_loaded = True
            except Exception:
                # Model will be loaded on actual use
                pass

    def _detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image.
        
        Returns:
            List of face detection results
        """
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )
            return faces
        except ValueError as e:
            if "Face could not be detected" in str(e):
                raise FaceNotDetectedException()
            raise

    def _generate_embedding(self, image: np.ndarray, use_largest_face: bool = True) -> List[float]:
        """
        Generate face embedding from image.
        
        Args:
            image: RGB numpy array
            use_largest_face: If True, use the largest face when multiple detected
            
        Returns:
            512-dimensional embedding vector
        """
        self._ensure_model_loaded()

        try:
            # DeepFace.represent returns list of dicts with 'embedding' key
            result = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )

            if not result:
                raise FaceNotDetectedException()

            if len(result) > 1:
                if use_largest_face:
                    # Find the largest face by facial area
                    largest_face = max(result, key=lambda x: self._get_face_area(x))
                    embedding = largest_face["embedding"]
                else:
                    raise MultipleFacesException()
            else:
                embedding = result[0]["embedding"]
            
            return embedding

        except ValueError as e:
            if "Face could not be detected" in str(e):
                raise FaceNotDetectedException()
            raise InvalidImageException(message=str(e))

    def _get_face_area(self, face_result: dict) -> float:
        """Calculate face area from detection result."""
        try:
            # DeepFace returns facial_area with x, y, w, h
            if "facial_area" in face_result:
                area = face_result["facial_area"]
                return area.get("w", 0) * area.get("h", 0)
            return 0
        except Exception:
            return 0

    async def detect_and_embed(self, image_bytes: bytes) -> List[float]:
        """
        Detect face and generate embedding from image bytes.
        
        This is the main method for processing uploaded images.
        Runs CPU-bound operations in thread pool.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            512-dimensional embedding vector
            
        Raises:
            FaceNotDetectedException: If no face detected
            MultipleFacesException: If multiple faces detected
            InvalidImageException: If image processing fails
        """
        # Convert bytes to numpy array
        image = bytes_to_numpy(image_bytes)

        # Run face detection and embedding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            _executor, self._generate_embedding, image
        )

        return embedding

    async def detect_faces_count(self, image_bytes: bytes) -> int:
        """
        Count faces in image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Number of faces detected
        """
        image = bytes_to_numpy(image_bytes)

        loop = asyncio.get_event_loop()
        try:
            faces = await loop.run_in_executor(
                _executor, self._detect_faces, image
            )
            return len(faces)
        except FaceNotDetectedException:
            return 0

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def distance_to_confidence(self, distance: float) -> float:
        """
        Convert cosine distance to confidence score.
        
        Cosine distance = 1 - cosine_similarity
        So confidence = 1 - distance = cosine_similarity
        
        Args:
            distance: Cosine distance (0-2, lower is more similar)
            
        Returns:
            Confidence score (0-1, higher is better)
        """
        # Clamp distance to valid range
        distance = max(0.0, min(2.0, distance))
        # Convert to confidence (similarity)
        confidence = 1.0 - distance
        # Clamp to 0-1 range
        return max(0.0, min(1.0, confidence))


# Singleton instance for reuse
_face_service_instance: Optional[FaceService] = None


def get_face_service() -> FaceService:
    """Get singleton face service instance."""
    global _face_service_instance
    if _face_service_instance is None:
        _face_service_instance = FaceService()
    return _face_service_instance
