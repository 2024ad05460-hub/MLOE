"""Shared TFLite runner: handles float32 and full-INT8 signatures transparently.

Used by convert_ptq.py, prune_quantise.py, benchmark.py, drift_monitor.py and
inference_service.py so that quantisation de/re-scaling is implemented exactly once.
"""

from pathlib import Path

import numpy as np

CLASS_NAMES = ["Normal", "Warning", "Critical"]

def _load_interpreter():
    """Prefer the 2 MB edge runtime; fall back to LiteRT, then to full TensorFlow.
    The Docker image on the truck installs tflite-runtime ONLY - the 600 MB
    tensorflow wheel never ships to a Raspberry Pi."""
    try:
        from tflite_runtime.interpreter import Interpreter        # edge device
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter        # TF >= 2.20
        return Interpreter
    except ImportError:
        import tensorflow as tf                                   # dev laptop
        return tf.lite.Interpreter


Interpreter = _load_interpreter()


class TFLiteModel:
    """Thin wrapper. predict() always takes float32 features and returns float
    probabilities, regardless of whether the model is FP32 or full-INT8."""

    def __init__(self, path):
        self.path = Path(path)
        self.interp = Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]
        self.int8 = self.inp["dtype"] == np.int8

    def size_kb(self):
        return self.path.stat().st_size / 1024.0

    def _quantise(self, x):
        """float32 features -> the tensor dtype the interpreter expects."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if not self.int8:
            return x
        q = np.round(x / self.in_scale + self.in_zp)
        return np.clip(q, -128, 127).astype(np.int8)

    def predict(self, x):
        """x: (6,) or (1,6) float32 normalised features -> (3,) probabilities."""
        self.interp.set_tensor(self.inp["index"], self._quantise(x))
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0]
        if self.int8:
            y = (y.astype(np.float32) - self.out_zp) * self.out_scale
        return y.astype(np.float32)


def evaluate(path, X, y):
    """Accuracy, per-class recall and confusion matrix on a held-out set."""
    m = TFLiteModel(path)
    pred = np.array([m.predict(x).argmax() for x in X])
    acc = float((pred == y).mean())
    recall = {CLASS_NAMES[c]: float((pred[y == c] == c).mean()) for c in range(3)}
    cm = [[int(((y == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)]
    return {"accuracy": acc, "recall": recall, "confusion": cm,
            "size_kb": m.size_kb()}
