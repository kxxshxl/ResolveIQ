"""Severity + sentiment from the complaint text itself.

Severity and sentiment are properties of *how the customer describes the impact*, not of the topic, so voting among
topic-similar past tickets (kNN) cannot recover them. This model scores a fixed set of semantic cues with a pretrained
NLI model ("The customer is losing income" -> entailment probability) and maps the cue vector to labels with a tiny
multinomial logistic combiner (see scripts/train_affect.py). The NLI model supplies the language understanding, so the
combiner can be trained on synthetic tickets and still generalise to hand-written text.

If the model or artifact is unavailable the classifier falls back to the rule/kNN ensemble, so this never blocks a request.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from app.classification.taxonomy import Taxonomy
from app.core.config import Settings
from app.observability.metrics import AFFECT_LATENCY

log = logging.getLogger(__name__)

# Cue hypotheses. Order matters (the artifact stores weights per cue name); a change invalidates the artifact (hash check).
CUES: dict[str, str] = {
    "angry": "The customer is angry, hostile or threatens to cancel or complain.",
    "repeated": "The customer has already tried several times to fix this without success.",
    "worried": "The customer is worried or anxious.",
    "calm": "The customer is calm and polite.",
    "exasperated": "The customer sounds exasperated and fed up.",
    "work": "The problem is affecting the customer's job, business, income or studies.",
    "safety": "There is a risk to health or safety, or an emergency.",
    "notur": "The customer says the issue is minor or not urgent.",
    "many": "Many people or a whole area are affected.",
    "urgent": "The customer needs the problem fixed urgently or today.",
    "total": "The customer has lost the service completely.",
    "days": "The problem has lasted for several days.",
    "money": "The customer is losing money or being wrongly charged.",
    "demand": "The customer demands compensation or immediate action.",
}


def cue_hash() -> str:
    return hashlib.sha256(json.dumps(CUES, sort_keys=True).encode()).hexdigest()[:16]


class AffectModel:
    name = "affect"

    def __init__(self, settings: Settings):
        self.s = settings
        self.model_name = settings.affect_model
        self.artifact_path = Path(settings.affect_artifact) if settings.affect_artifact else settings.data_dir / "models" / "affect_v1.json"
        self.artifact: dict | None = None
        self.available = False
        self._tok = self._model = None
        self._entail_idx = 0
        self._device = "cpu"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load(self, require_artifact: bool = True) -> None:
        """Load the NLI model (and the trained combiner). `require_artifact=False` is used by the training script."""
        if not self.s.affect_enabled:
            log.info("affect model disabled by configuration")
            return
        try:
            if require_artifact:
                self.artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
                if self.artifact.get("model") != self.model_name or self.artifact.get("cue_hash") != cue_hash():
                    raise ValueError("artifact was trained for a different NLI model or cue set; retrain with scripts/train_affect.py")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            t0 = time.perf_counter()
            self._device = ("cuda" if torch.cuda.is_available() else "cpu") if self.s.affect_device == "auto" else self.s.affect_device
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            # The checkpoint is fp16; recent transformers keep that dtype, which is ~10x slower on CPU than fp32.
            dtype = torch.float32 if self._device == "cpu" else torch.float16
            try:
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, dtype=dtype).eval()
            except TypeError:  # transformers < 4.56 spells it torch_dtype
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=dtype).eval()
            if self._device == "cpu":
                torch.set_num_threads(max(1, self.s.affect_cpu_threads))  # NB: int8 dynamic quantisation collapses DeBERTa accuracy (see docs)
            self._model = model.to(self._device)
            self._entail_idx = next(i for i, l in model.config.id2label.items() if l.lower().startswith("entail"))
            self.available = True
            self.cue_scores(["warmup"])
            log.info("affect model loaded", extra={"model": self.model_name, "device": self._device,
                                                   "seconds": round(time.perf_counter() - t0, 2)})
        except Exception as exc:  # noqa: BLE001 - optional component: degrade to the rule/kNN ensemble
            self.available = False
            log.warning("affect model unavailable; using rule/kNN fallback for severity and sentiment",
                        extra={"error": f"{type(exc).__name__}: {exc}"[:300]})

    # ------------------------------------------------------------------ inference
    def cue_scores(self, texts: list[str], batch: int = 8) -> np.ndarray:
        import torch

        hyps = list(CUES.values())
        out = np.zeros((len(texts), len(hyps)), dtype=np.float32)
        with self._lock, torch.no_grad():
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                prem = [t for t in chunk for _ in hyps]
                hyp = [h for _ in chunk for h in hyps]
                enc = self._tok(prem, hyp, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.s.affect_max_tokens).to(self._device)
                probs = torch.softmax(self._model(**enc).logits, -1)[:, self._entail_idx]
                out[i:i + len(chunk)] = probs.float().cpu().numpy().reshape(len(chunk), len(hyps))
        return out

    def combine(self, x: np.ndarray) -> list[dict[str, dict[str, float]]]:
        res: list[dict[str, dict[str, float]]] = [{} for _ in range(len(x))]
        for dim, spec in (self.artifact or {}).get("dimensions", {}).items():
            z = x @ np.asarray(spec["coef"], dtype=np.float64).T + np.asarray(spec["intercept"])
            z = np.exp(z - z.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
            for r, row in zip(res, p):
                r[dim] = {c: float(v) for c, v in zip(spec["classes"], row)}
        return res

    def predict_sync(self, texts: list[str]) -> list[dict[str, dict[str, float]]]:
        t0 = time.perf_counter()
        out = self.combine(self.cue_scores(texts))
        AFFECT_LATENCY.observe(time.perf_counter() - t0)
        return out

    # Classifier protocol (see strategies.py): only severity and sentiment are produced here.
    async def predict(self, ctx, tax: Taxonomy) -> dict[str, dict[str, float]]:
        dist: dict[str, dict[str, float]] = {"intent": {}, "product": {}, "severity": {}, "sentiment": {}}
        if not self.available:
            return dist
        probs = (await asyncio.to_thread(self.predict_sync, [ctx.text]))[0]
        for dim in ("severity", "sentiment"):
            dist[dim] = {k: v for k, v in probs.get(dim, {}).items() if tax.has(dim, k)}
        return dist
