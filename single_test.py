from server import Server
from client import Client
import numpy as np
import calculater.calculater_pb2 as pb
from utils import *

if __name__ == "__main__":
    server_calc = Server("./config.yaml")
    client_calc = Client("./config.yaml")
    start_pos = 0

    name_weight_d = server_calc.init_model()
    tokens, start_pos = server_calc.encode(input_lst=["Hello?"], start_pos=start_pos)
    print(tokens)
    h, start_pos, freqs_cis, mask = server_calc.pre_process(tokens, start_pos)
    print(h.shape, start_pos, freqs_cis.shape, mask.shape)

    print(name_weight_d.keys())
    key = "wq0"
    client_calc.load({key: name_weight_d[key]})
    xq = client_calc.proj(h, "wq0")
    print(xq.shape)

    xq, xk, xv = server_calc.rotary_emb(h, 0, xq, xq, xq, freqs_cis, start_pos)
    print(xq.shape, xk.shape, xv.shape)
    # print(convert_bsztensor_np2pb(xq))

    key = "w10"
    client_calc.load({key: name_weight_d[key]})
    h1 = client_calc.feed_forward(h, "w10")
    print(h1.shape)

    key = "out"
    client_calc.load({key: name_weight_d[key]})
    output = client_calc.out(h)
    print(output.shape)

    key = "wo0"
    client_calc.load({key: name_weight_d[key]})
    bsz, seqlen, _ = h.shape
    h = client_calc.score(xq, xk, xv, mask, bsz, seqlen, "wo0")
    print(h.shape)
