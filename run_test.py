import grpc
import numpy as np

# 导入编译后的Protobuf消息类
import calculater.calculater_pb2 as pb
import calculater.calculater_pb2_grpc as pb_grpc
from utils import *
import copy

MAX_MESSAGE_LENGTH = 500 * 1024 * 1024
options = [
    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.enable_retries", 1),
]


def print_model_response(response):
    for matrix in response.matrix:
        print("Matrix Name:", matrix.name)
        # print("Matrix Data:", matrix.data)


def run():
    channel_server = grpc.insecure_channel(
        "localhost:50051", options=copy.deepcopy(options)
    )
    channel_client = grpc.insecure_channel(
        "localhost:50052", options=copy.deepcopy(options)
    )
    # with grpc.insecure_channel("localhost:50051", options=options) as channel:
    if True:
        stub_server = pb_grpc.CalculaterStub(channel_server)
        stub_client = pb_grpc.CalculaterStub(channel_client)

        # 创建包含NumPy数组的结构体
        response_w = stub_server.GetModelByPath(pb.GetModelByPathRequest())
        # print_model_response(response)

        response = stub_server.Encode(pb.EncodeRequest(start_pos=0, input=["Hello?"]))

        tokens, start_pos = response.tokens, response.start_pos
        print(tokens, start_pos, convert_matrix_pb2np(tokens))
        response = stub_server.PreProcess(
            pb.PreProcessRequest(tokens=tokens, start_pos=start_pos)
        )
        # print(response.h, response.start_pos, response.freqs_cis, response.mask)

        for matrix in response_w.matrix:
            if matrix.name == "wq0":
                response_load = stub_client.LoadModel(
                    pb.LoadModelRequest(name=matrix.name, data=matrix.data)
                )
                print(response_load, "-" * 10)
            if matrix.name == "w10":
                response_load = stub_client.LoadModel(
                    pb.LoadModelRequest(name=matrix.name, data=matrix.data)
                )
                print(response_load, "-" * 10)
            if matrix.name == "wo0":
                response_load = stub_client.LoadModel(
                    pb.LoadModelRequest(name=matrix.name, data=matrix.data)
                )
                print(response_load, "-" * 10)
            if matrix.name == "out":
                print(len(matrix.data))
                response_load = stub_client.LoadModel(
                    pb.LoadModelRequest(name=matrix.name, data=matrix.data)
                )
                print(response_load, "-" * 10)

        response_proj = stub_client.Proj(pb.ProjRequest(name="wq0", h=response.h))
        print(response_proj)

        response_qkv = stub_server.RotrayEmb(
            pb.RotrayEmbRequest(
                h=response.h,
                i=0,
                xq=response_proj.x,
                xk=response_proj.x,
                xv=response_proj.x,
                freqs_cis=response.freqs_cis,
                start_pos=response.start_pos,
            )
        )
        # print(response_qkv)
        print("qkv", "-" * 10)

        response_feedword = stub_client.FeedForward(
            pb.FeedForwardRequest(h=response.h, name="w10")
        )
        print("feedword", "-" * 10)

        response = stub_client.Score(
            pb.ScoreRequest(
                xq=response_qkv.xq,
                xk=response_qkv.xk,
                xv=response_qkv.xv,
                mask=response.mask,
                bsz=1,
                seqlen=3,
                name="wo0",
            )
        )
        print(response)

        response = stub_client.Out(pb.OutRequest(h=response.h))
        print(response)


if __name__ == "__main__":
    run()
