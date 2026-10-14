from typing import List, Tuple, Optional
import numpy as np
import torch
import open_clip
from PIL import Image


class CLIPGrounder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_image(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame[:, :, ::-1])
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    def ground(
        self,
        frame: np.ndarray,
        detections: List[dict],
        query: str,
        top_k: int = 1,
    ) -> List[dict]:
        if not detections:
            return []
        text_feat = self.encode_text(query)
        scored = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            img_feat = self.encode_image(crop)
            score = float(np.dot(img_feat, text_feat.T).squeeze())
            item = dict(det)
            item["ground_score"] = score
            item["query"] = query
            scored.append(item)
        scored.sort(key=lambda x: x["ground_score"], reverse=True)
        return scored[:top_k]
