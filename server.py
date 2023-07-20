import numpy as np
from typing import Any, Tuple, Optional, List, Dict, Union
from llama import Tokenizer
import pickle
import yaml
from llama.model_numpy import EmbeddingNumpy, apply_rotary_emb, TransformerNumpy
import copy


class Server:
    def __init__(self, config_fname: str) -> None:
        with open(config_fname, "r") as file:
            config = yaml.safe_load(file)

        self.tokenizer = Tokenizer(model_path=config["tokenizer_path"])
        # 暂定只有一个
        with open(config["model_path"][0], "rb") as f:
            self.model = pickle.load(f)
        self.tok_embeddings = EmbeddingNumpy(self.model.tok_embeddings.weights)
        self.freqs_cis = self.model.freqs_cis

        self.bsz, self.seqlen = config["bsz"], config["seqlen"]
        self.n_local_heads, self.head_dim = (
            config["n_heads"],
            config["dim"] // config["n_heads"],
        )
        self.max_batch_size, self.max_seq_len = (
            config["max_batch_size"],
            config["max_seq_len"],
        )

    def init_model(self):
        weight_d = {
            "out": {
                "mlp": self.model.output.weights,
                "norm": [self.model.norm.weight, self.model.norm.eps],
            }
        }
        cache_ori = np.zeros(
            (self.max_batch_size, self.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_k_lst = [
            copy.deepcopy(cache_ori) for _ in range(len(self.model.layers))
        ]
        self.cache_v_lst = [
            copy.deepcopy(cache_ori) for _ in range(len(self.model.layers))
        ]
        for i, layer in enumerate(self.model.layers):
            weight_d[f"wq{i}"] = {
                "mlp": layer.attention.wq.weights,
                "norm": [layer.attention_norm.weight, layer.attention_norm.eps],
            }
            weight_d[f"wk{i}"] = {
                "mlp": layer.attention.wk.weights,
                "norm": [layer.attention_norm.weight, layer.attention_norm.eps],
            }
            weight_d[f"wv{i}"] = {
                "mlp": layer.attention.wv.weights,
                "norm": [layer.attention_norm.weight, layer.attention_norm.eps],
            }
            weight_d[f"wo{i}"] = {"mlp": layer.attention.wo.weights}
            weight_d[f"w1{i}"] = {
                "mlp": layer.feed_forward.w1.weights,
                "norm": [layer.ffn_norm.weight, layer.ffn_norm.eps],
            }
            weight_d[f"w2{i}"] = {"mlp": layer.feed_forward.w2.weights}
            weight_d[f"w3{i}"] = {
                "mlp": layer.feed_forward.w3.weights,
                "norm": [layer.ffn_norm.weight, layer.ffn_norm.eps],
            }
        return weight_d

    def encode(
        self, input_lst: List[str], start_pos: int = 0
    ) -> Tuple[np.ndarray[int], int]:
        tokens = [
            self.tokenizer.encode(input_, bos=True, eos=False) for input_ in input_lst
        ]
        return np.array(tokens), start_pos

    def decode(self, lst: np.ndarray[int]):
        pass

    def pre_process(
        self, tokens: np.ndarray, start_pos: int
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        h = self.tok_embeddings(tokens)

        _bsz, seqlen = tokens.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = np.full((1, seqlen, seqlen), float("-inf"))
            mask = np.triu(mask.squeeze(), k=start_pos + 1)
        else:
            mask = np.zeros((1, seqlen, seqlen))

        return h, start_pos, freqs_cis, mask

    def post_process(self, h: np.ndarray[np.float32]):
        pass

    def rotary_emb(
        self,
        h: np.ndarray,
        i: int,
        xq: np.ndarray,
        xk: np.ndarray,
        xv: np.ndarray,
        freqs_cis: np.ndarray,
        start_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # i: 第多少层
        bsz, seqlen, _ = h.shape

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k_lst[i][: self.bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v_lst[i][: self.bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k_lst[i][:bsz, : start_pos + seqlen]
        values = self.cache_v_lst[i][:bsz, : start_pos + seqlen]

        return (
            xq.transpose((0, 2, 1, 3)),
            keys.transpose((0, 2, 1, 3)),
            values.transpose((0, 2, 1, 3)),
        )
