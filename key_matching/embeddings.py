from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class EmbeddingExtractor:
    """Lazy TorchScript ResNet18 embedding inference."""

    def __init__(self, model_path: str | None) -> None:
        self.model_path = model_path
        self._model = None

    def _load(self):
        if self._model is None and self.model_path and Path(self.model_path).exists():
            import torch
            self._model = torch.jit.load(self.model_path, map_location="cpu").eval()
        return self._model

    def extract(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        model = self._load()
        if model is None:
            return np.empty(0, dtype=np.float32)
        import torch
        foreground = cv2.bitwise_and(image, image, mask=mask)
        rgb = cv2.cvtColor(cv2.resize(foreground, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        with torch.inference_mode():
            output = model(((tensor - mean) / std).unsqueeze(0))[0]
        vector = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
