import grpc
import argparse
from concurrent import futures
import calculater.calculater_pb2_grpc as pb_grpc
import calculater.calculater_pb2 as pb
import pickle
from grpc import experimental
import numpy as np
from typing import Union
from utils import *

# 增加消息大小限制
# 客户端数据传输大小配置
MAX_MESSAGE_LENGTH = 500 * 1024 * 1024

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()  # 添加两个互斥的参数选项
group.add_argument("--server", action="store_true", help="run server")
group.add_argument("--client", action="store_true", help="run client")
parser.add_argument("--port", default=50051, type=int, help="run port")


class MyService(pb_grpc.CalculaterServicer):
    def __init__(self, calc):
        self.calc = calc
        super().__init__()

    def GetModelByPath(self, request, context):
        weight_d = self.calc.init_model()
        response = pb.GetModelByPathResponse()

        # 创建并填充CombinedMatrix对象
        for k, v in weight_d.items():
            combined_matrix = response.matrix.add()
            combined_matrix.name = k
            combined_matrix.data = pickle.dumps(v)
        return response

    def Encode(self, request, context):
        tokens, start_pos = self.calc.encode(
            input_lst=request.input, start_pos=request.start_pos
        )
        return pb.EncodeResponse(
            start_pos=start_pos, tokens=convert_matrix_np2pb(tokens, True)
        )

    def PreProcess(self, request, context):
        h, start_pos, freqs_cis, mask = self.calc.pre_process(
            convert_matrix_pb2np(request.tokens), request.start_pos
        )
        return pb.PreProcessResponse(
            start_pos=start_pos,
            h=convert_tensor_np2pb(h),
            freqs_cis=convert_matrix_np2pb(freqs_cis, is_complex=True),
            mask=convert_matrix_np2pb(mask),
        )

    def RotrayEmb(self, request, context):
        xq, xk, xv = self.calc.rotary_emb(
            h=convert_tensor_pb2np(request.h),
            i=request.i,
            xq=convert_bsztensor_pb2np(request.xq),
            xk=convert_bsztensor_pb2np(request.xk),
            xv=convert_bsztensor_pb2np(request.xv),
            freqs_cis=convert_matrix_pb2np(request.freqs_cis, is_complex=True),
            start_pos=request.start_pos,
        )
        return pb.RotrayEmbResponse(
            xq=convert_bsztensor_np2pb(xq),
            xk=convert_bsztensor_np2pb(xk),
            xv=convert_bsztensor_np2pb(xv),
        )

    def LoadModel(self, request, context):
        self.calc.load({request.name: pickle.loads(request.data)})
        return pb.LoadModelResponse()

    def Proj(self, request, context):
        return pb.ProjResponse(
            x=convert_bsztensor_np2pb(
                self.calc.proj(h=convert_tensor_pb2np(request.h), name=request.name)
            )
        )

    def Score(self, request, context):
        return pb.ScoreResponse(
            h=convert_tensor_np2pb(
                self.calc.score(
                    q=convert_bsztensor_pb2np(request.xq),
                    k=convert_bsztensor_pb2np(request.xk),
                    v=convert_bsztensor_pb2np(request.xv),
                    mask=convert_matrix_pb2np(request.mask),
                    bsz=request.bsz,
                    seqlen=request.seqlen,
                    name=request.name,
                )
            )
        )

    def FeedForward(self, request, context):
        return pb.FeedForwardResponse(
            h=convert_tensor_np2pb(
                self.calc.feed_forward(
                    convert_tensor_pb2np(request.h), name=request.name
                )
            )
        )

    def Out(self, request, context):
        return pb.OutResponse(
            output=convert_matrix_np2pb(self.calc.out(convert_tensor_pb2np(request.h)))
        )


def run(MyService, calc, port=50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    pb_grpc.add_CalculaterServicer_to_server(MyService(calc), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.server:
        from server import Server

        calc = Server("./config.yaml")
        run(MyService, calc, args.port)
    elif args.client:
        from client import Client

        calc = Client("./config.yaml")
        run(MyService, calc, args.port + 1)
    else:
        print("Please specify either --server or --client option.")
        exit()
