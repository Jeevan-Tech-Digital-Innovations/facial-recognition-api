import asyncio
import logging
import math
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
from app.utils.image_utils import bytes_to_numpy, resize_image

logger = logging.getLogger(__name__)

settings = get_settings()

# Thread pool for CPU-bound face operations
_executor = ThreadPoolExecutor(max_workers=4)

# Minimum detector confidence to accept a face as real.
# Reject low-confidence detections that produce garbage embeddings.
MIN_FACE_CONFIDENCE = 0.90


def _l2_normalize(embedding: List[float]) -> List[float]:
    """
    L2-normalize an embedding to a unit vector.

    ArcFace is trained to produce unit vectors, but we enforce this
    explicitly so that cosine distance in pgvector is always correct
    regardless of minor floating-point drift or model changes.
    """
    vec = np.array(embedding, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


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
                logger.info("Loading face recognition model: %s", self.model_name)
                dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
                DeepFace.represent(
                    dummy_img,
                    model_name=self.model_name,
                    detector_backend="skip",  # Skip detection for dummy
                    enforce_detection=False,
                )
                self._model_loaded = True
                logger.info("Face recognition model loaded successfully")
            except Exception:
                logger.warning("Model pre-load failed; will load on first use")

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

        Pipeline:
        1. Detect face(s) via DeepFace.represent
        2. Filter by detector confidence (reject low-quality detections)
        3. Select single face (largest or reject if multiple)
        4. L2-normalize the embedding to a unit vector

        Args:
            image: RGB numpy array
            use_largest_face: If True, use the largest face when multiple detected

        Returns:
            512-dimensional L2-normalized embedding vector
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

            # Filter out low-confidence detections to prevent garbage
            # embeddings from polluting the database
            confident_faces = [
                r for r in result
                if r.get("face_confidence", 0) >= MIN_FACE_CONFIDENCE
            ]

            if not confident_faces:
                best_conf = max(r.get("face_confidence", 0) for r in result)
                logger.warning(
                    "All detected faces below confidence threshold: "
                    "best=%.2f, required=%.2f",
                    best_conf, MIN_FACE_CONFIDENCE,
                )
                raise FaceNotDetectedException(
                    message="Face detected but confidence too low. "
                    "Please try with better lighting or a clearer image.",
                )

            if len(confident_faces) > 1:
                logger.info("Multiple confident faces detected: count=%d", len(confident_faces))
                if use_largest_face:
                    # Find the largest face by facial area
                    chosen = max(confident_faces, key=lambda x: self._get_face_area(x))
                else:
                    raise MultipleFacesException()
            else:
                chosen = confident_faces[0]

            embedding = chosen["embedding"]

            # L2-normalize to guarantee unit vector for cosine distance
            return _l2_normalize(embedding)

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

        Pipeline: bytes -> EXIF-corrected numpy -> resize -> detect -> embed -> normalize

        Args:
            image_bytes: Raw image bytes

        Returns:
            512-dimensional L2-normalized embedding vector

        Raises:
            FaceNotDetectedException: If no face detected or confidence too low
            MultipleFacesException: If multiple faces detected
            InvalidImageException: If image processing fails
        """
        # Convert bytes to numpy array (EXIF orientation is applied inside)
        image = bytes_to_numpy(image_bytes)

        # Pre-resize large images to save memory and speed up detection.
        # Faces don't need more than ~640px on the longest side.
        image = resize_image(image)

        # Run face detection and embedding in thread pool
        loop = asyncio.get_running_loop()
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
        image = resize_image(image)

        loop = asyncio.get_running_loop()
        try:
            faces = await loop.run_in_executor(
                _executor, self._detect_faces, image
            )
            return len(faces)
        except FaceNotDetectedException:
            return 0

    def distance_to_confidence(self, distance: float) -> float:
        """
        Convert cosine distance to a human-intuitive confidence score
        using a sigmoid curve centered around the match threshold.

        The sigmoid produces:
        - ~95% confidence for distance ≈ 0.10  (strong match)
        - ~50% confidence for distance ≈ threshold  (borderline)
        - ~5%  confidence for distance ≈ 1.0   (non-match)

        This is much more meaningful than the naive linear mapping
        ``confidence = 1 - distance`` which made 0.30 (solid match)
        report as only 70%.

        Args:
            distance: Cosine distance (0-2, lower is more similar)

        Returns:
            Confidence score (0-1, higher is better)
        """
        distance = max(0.0, min(2.0, distance))
        # Steepness factor -- controls how sharply confidence drops
        k = 10.0
        midpoint = self.threshold
        confidence = 1.0 / (1.0 + math.exp(k * (distance - midpoint)))
        return round(confidence, 4)


# Singleton instance for reuse
_face_service_instance: Optional[FaceService] = None


def get_face_service() -> FaceService:
    """Get singleton face service instance."""
    global _face_service_instance
    if _face_service_instance is None:
        _face_service_instance = FaceService()
    return _face_service_instance
