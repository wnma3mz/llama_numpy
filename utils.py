import numpy as np
from typing import Union
import calculater.calculater_pb2 as pb


def convert_matrix_pb2np(
    matrix: Union[pb.Matrix, pb.MatrixInt, pb.MatrixComplex], is_complex: bool = False
) -> np.ndarray:
    if is_complex:
        real_part = np.array([array.value for array in matrix.real_array])
        imag_part = np.array([array.value for array in matrix.imag_array])
        return np.vectorize(complex)(real_part, imag_part)
    return np.array([array.value for array in matrix.array])


def convert_matrix_np2pb(
    matrix: np.ndarray, is_int: bool = False, is_complex: bool = False
) -> Union[pb.Matrix, pb.MatrixInt, pb.MatrixComplex]:
    if is_int:
        return pb.MatrixInt(array=[pb.ArrayInt(value=array) for array in matrix])
    if is_complex:
        return pb.MatrixComplex(
            real_array=[pb.Array(value=array) for array in matrix.real],
            imag_array=[pb.Array(value=array) for array in matrix.imag],
        )
    return pb.Matrix(array=[pb.Array(value=array) for array in matrix])


def convert_tensor_pb2np(tensor: pb.Tensor) -> np.ndarray:
    return np.array(
        [[array.value for array in matrix.array] for matrix in tensor.matrix]
    )


def convert_tensor_np2pb(tensor: np.ndarray) -> pb.Tensor:
    return pb.Tensor(
        matrix=[
            pb.Matrix(array=[pb.Array(value=array) for array in matrix])
            for matrix in tensor
        ]
    )


def convert_bsztensor_pb2np(bzstensor: pb.BszTensor) -> np.ndarray:
    return np.array(
        [
            [[array.value for array in matrix.array] for matrix in tensor.matrix]
            for tensor in bzstensor.tensor
        ]
    )


def convert_bsztensor_np2pb(bzstensor: np.ndarray) -> pb.BszTensor:
    return pb.BszTensor(
        tensor=[
            pb.Tensor(
                matrix=[
                    pb.Matrix(array=[pb.Array(value=array) for array in matrix])
                    for matrix in tensor
                ]
            )
            for tensor in bzstensor
        ]
    )


if __name__ == "__main__":
    # 测试上述所有的函数
    # 测试convert_matrix_pb2np
    matrix = pb.Matrix(array=[pb.Array(value=[1, 2, 3]), pb.Array(value=[4, 5, 6])])
    print(convert_matrix_pb2np(matrix))
    print(convert_matrix_np2pb(convert_matrix_pb2np(matrix), is_int=False))
    print("=" * 20)
    # 测试convert_tensor_pb2np
    tensor = pb.Tensor(
        matrix=[
            pb.Matrix(array=[pb.Array(value=[1, 2, 3]), pb.Array(value=[4, 5, 6])]),
            pb.Matrix(array=[pb.Array(value=[7, 8, 9]), pb.Array(value=[10, 11, 12])]),
        ]
    )
    print(convert_tensor_pb2np(tensor))
    print(convert_tensor_np2pb(convert_tensor_pb2np(tensor)))
    print("=" * 20)

    # 测试convert_bsztensor_pb2np
    bzstensor = pb.BszTensor(
        tensor=[
            pb.Tensor(
                matrix=[
                    pb.Matrix(
                        array=[pb.Array(value=[1, 2, 3]), pb.Array(value=[4, 5, 6])]
                    )
                ]
            )
        ]
    )
    print(convert_bsztensor_pb2np(bzstensor))
    print(convert_bsztensor_np2pb(convert_bsztensor_pb2np(bzstensor)))
    print("=" * 20)
