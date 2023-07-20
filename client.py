import numpy as np
from typing import Any, Tuple, Optional, List, Dict
from llama import Tokenizer
import yaml
from llama.model_numpy import RMSNormNumpy, LinearNumpy, softmax


class Client:
    def __init__(self, config_fname: str = "config.yaml") -> None:
        with open(config_fname, "r") as file:
            config = yaml.safe_load(file)
        self.n_local_heads, self.head_dim = (
            config["n_heads"],
            config["dim"] // config["n_heads"],
        )

    def load(self, name_weight_d: Dict[str, np.ndarray]) -> None:
        # 0.
        for name, weight in name_weight_d.items():
            if name == "out":
                self.output = LinearNumpy(weight["mlp"])
                self.norm = RMSNormNumpy(*weight["norm"])
            for key in ["wq", "wk", "wv", "w1", "w3"]:
                if key in name:
                    setattr(self, f"{name}.mlp", LinearNumpy(weight["mlp"]))
                    setattr(self, f"{name}.norm", RMSNormNumpy(*weight["norm"]))
            for key in ["wo", "w3"]:
                if key in name:
                    setattr(self, f"{name}.mlp", LinearNumpy(weight["mlp"]))

    def proj(self, h: np.ndarray, name: str) -> np.ndarray:
        # name: 对应使用的层
        # 1.
        bsz, seqlen, _ = h.shape
        return getattr(self, f"{name}.mlp")(getattr(self, f"{name}.norm")(h)).reshape(
            (bsz, seqlen, self.n_local_heads, self.head_dim)
        )

    def score(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray],
        bsz: int,
        seqlen: int,
        name: str,
    ) -> np.ndarray:
        # name: 对应使用的层
        # 2.
        scores = np.matmul(q, k.transpose((0, 1, 3, 2))) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = softmax(scores.astype(np.float32), dim=-1).astype(np.float32)
        output = np.matmul(scores, v)  # (bs, n_local_heads, slen, head_dim)
        # output = contiguous(output.transpose((0, 2, 1, 3))).reshape(bsz, seqlen, -1)
        output = output.transpose((0, 2, 1, 3)).reshape(bsz, seqlen, -1)
        return getattr(self, f"{name}.mlp")(output)

    def feed_forward(self, h: np.ndarray, name: str) -> np.ndarray:
        # 3.
        if f"{name}.norm" in self.__dict__:
            return getattr(self, f"{name}.mlp")(getattr(self, f"{name}.norm")(h))
        else:
            return getattr(self, f"{name}.mlp")(h)

    def out(self, h: np.ndarray) -> np.ndarray:
        # 4.
        return self.output(self.norm(h)[:, -1, :])  # only compute last logits
