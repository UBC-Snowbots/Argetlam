"""Open-vocabulary (text-promptable) object detector backends.

Backends (lazy-imported so the control host `.venv` needs none of this unless camera perception is
actually used - run on a python with the `vision` extra / torch, e.g. the Isaac env):
  * "owlv2"  - google/owlv2-base-patch16-ensemble via transformers (zero-shot, prompt per class).
  * "gdino"  - IDEA-Research/grounding-dino-tiny via transformers (one text query, many boxes).

Each `detect(image_rgb, prompts, min_conf)` returns a list of (label, (u0, v0, u1, v1), score) with
pixel boxes.  `label` is the matched prompt text; the caller maps it to a known object name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RawDetection:
    label: str
    bbox: tuple[float, float, float, float]
    score: float


class Detector:
    def __init__(self, backend: str = "owlv2", model: str | None = None, device: str = "cuda",
                 dtype: str = "float32"):
        self.backend = backend
        self.device = device
        self._model = None
        self._proc = None
        self._dtype = dtype
        self.model_id = model or {
            "owlv2": "google/owlv2-base-patch16-ensemble",
            "gdino": "IDEA-Research/grounding-dino-tiny",
        }[backend]

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        dt = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[self._dtype]
        # owlv2's fast CLIP tokenizer is incompatible with some tokenizers builds (RobertaProcessing
        # 'cls' kwarg); the slow tokenizer is fine and the text is tiny, so speed is irrelevant.
        proc_kw = {"use_fast": False} if self.backend == "owlv2" else {}
        self._proc = AutoProcessor.from_pretrained(self.model_id, **proc_kw)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, torch_dtype=dt)
        self._model.to(self.device).eval()

    def detect(self, image_rgb, prompts: list[str], min_conf: float = 0.1) -> list[RawDetection]:
        import torch
        from PIL import Image
        self.load()
        img = image_rgb if isinstance(image_rgb, Image.Image) else Image.fromarray(image_rgb)
        mdtype = next(self._model.parameters()).dtype

        def _cast(inp):
            # BatchFeature.to(device) moves everything; .to(dtype) casts only floating tensors
            # (pixel_values), leaving int tensors (input_ids / masks) alone - fixes the fp16 model /
            # fp32 pixel_values mismatch.
            return inp.to(self.device).to(mdtype)

        if self.backend == "gdino":
            text = ". ".join(p.lower() for p in prompts) + "."
            inputs = _cast(self._proc(images=img, text=text, return_tensors="pt"))
            with torch.no_grad():
                out = self._model(**inputs)
            res = self._proc.post_process_grounded_object_detection(
                out, threshold=min_conf, target_sizes=[img.size[::-1]])[0]
            labels = res.get("labels") or res.get("text_labels")
            return [RawDetection(str(lb), tuple(float(v) for v in box), float(sc))
                    for box, sc, lb in zip(res["boxes"].tolist(), res["scores"].tolist(), labels)]
        # owlv2: one query set, per-box label index
        inputs = _cast(self._proc(images=img, text=[prompts], return_tensors="pt"))
        with torch.no_grad():
            out = self._model(**inputs)
        res = self._proc.post_process_grounded_object_detection(
            out, threshold=min_conf, target_sizes=torch.tensor([img.size[::-1]]).to(self.device))[0]
        dets = []
        for box, sc, lb in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
            dets.append(RawDetection(prompts[int(lb)], tuple(float(v) for v in box), float(sc)))
        return dets

    def unload(self):
        import gc
        self._model = None
        self._proc = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
