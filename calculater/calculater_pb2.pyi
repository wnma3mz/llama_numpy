from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class Array(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, value: _Optional[_Iterable[float]] = ...) -> None: ...

class ArrayInt(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, value: _Optional[_Iterable[int]] = ...) -> None: ...

class Matrix(_message.Message):
    __slots__ = ["array"]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    array: _containers.RepeatedCompositeFieldContainer[Array]
    def __init__(
        self, array: _Optional[_Iterable[_Union[Array, _Mapping]]] = ...
    ) -> None: ...

class MatrixInt(_message.Message):
    __slots__ = ["array"]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    array: _containers.RepeatedCompositeFieldContainer[ArrayInt]
    def __init__(
        self, array: _Optional[_Iterable[_Union[ArrayInt, _Mapping]]] = ...
    ) -> None: ...

class MatrixComplex(_message.Message):
    __slots__ = ["real_array", "imag_array"]
    REAL_ARRAY_FIELD_NUMBER: _ClassVar[int]
    IMAG_ARRAY_FIELD_NUMBER: _ClassVar[int]
    real_array: _containers.RepeatedCompositeFieldContainer[Array]
    imag_array: _containers.RepeatedCompositeFieldContainer[Array]
    def __init__(
        self,
        real_array: _Optional[_Iterable[_Union[Array, _Mapping]]] = ...,
        imag_array: _Optional[_Iterable[_Union[Array, _Mapping]]] = ...,
    ) -> None: ...

class Tensor(_message.Message):
    __slots__ = ["matrix"]
    MATRIX_FIELD_NUMBER: _ClassVar[int]
    matrix: _containers.RepeatedCompositeFieldContainer[Matrix]
    def __init__(
        self, matrix: _Optional[_Iterable[_Union[Matrix, _Mapping]]] = ...
    ) -> None: ...

class BszTensor(_message.Message):
    __slots__ = ["tensor"]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    tensor: _containers.RepeatedCompositeFieldContainer[Tensor]
    def __init__(
        self, tensor: _Optional[_Iterable[_Union[Tensor, _Mapping]]] = ...
    ) -> None: ...

class GetModelByPathRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetModelByPathResponse(_message.Message):
    __slots__ = ["matrix"]
    MATRIX_FIELD_NUMBER: _ClassVar[int]
    matrix: _containers.RepeatedCompositeFieldContainer[CombinedMatrix]
    def __init__(
        self, matrix: _Optional[_Iterable[_Union[CombinedMatrix, _Mapping]]] = ...
    ) -> None: ...

class EncodeRequest(_message.Message):
    __slots__ = ["start_pos", "input"]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    start_pos: int
    input: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self, start_pos: _Optional[int] = ..., input: _Optional[_Iterable[str]] = ...
    ) -> None: ...

class EncodeResponse(_message.Message):
    __slots__ = ["start_pos", "tokens"]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    start_pos: int
    tokens: MatrixInt
    def __init__(
        self,
        start_pos: _Optional[int] = ...,
        tokens: _Optional[_Union[MatrixInt, _Mapping]] = ...,
    ) -> None: ...

class CombinedMatrix(_message.Message):
    __slots__ = ["name", "data"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    data: bytes
    def __init__(
        self, name: _Optional[str] = ..., data: _Optional[bytes] = ...
    ) -> None: ...

class PreProcessRequest(_message.Message):
    __slots__ = ["start_pos", "tokens"]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    start_pos: int
    tokens: MatrixInt
    def __init__(
        self,
        start_pos: _Optional[int] = ...,
        tokens: _Optional[_Union[MatrixInt, _Mapping]] = ...,
    ) -> None: ...

class PreProcessResponse(_message.Message):
    __slots__ = ["start_pos", "h", "freqs_cis", "mask"]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    FREQS_CIS_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    start_pos: int
    h: Tensor
    freqs_cis: MatrixComplex
    mask: Matrix
    def __init__(
        self,
        start_pos: _Optional[int] = ...,
        h: _Optional[_Union[Tensor, _Mapping]] = ...,
        freqs_cis: _Optional[_Union[MatrixComplex, _Mapping]] = ...,
        mask: _Optional[_Union[Matrix, _Mapping]] = ...,
    ) -> None: ...

class RotrayEmbRequest(_message.Message):
    __slots__ = ["i", "start_pos", "h", "xq", "xk", "xv", "freqs_cis"]
    I_FIELD_NUMBER: _ClassVar[int]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    XQ_FIELD_NUMBER: _ClassVar[int]
    XK_FIELD_NUMBER: _ClassVar[int]
    XV_FIELD_NUMBER: _ClassVar[int]
    FREQS_CIS_FIELD_NUMBER: _ClassVar[int]
    i: int
    start_pos: int
    h: Tensor
    xq: BszTensor
    xk: BszTensor
    xv: BszTensor
    freqs_cis: MatrixComplex
    def __init__(
        self,
        i: _Optional[int] = ...,
        start_pos: _Optional[int] = ...,
        h: _Optional[_Union[Tensor, _Mapping]] = ...,
        xq: _Optional[_Union[BszTensor, _Mapping]] = ...,
        xk: _Optional[_Union[BszTensor, _Mapping]] = ...,
        xv: _Optional[_Union[BszTensor, _Mapping]] = ...,
        freqs_cis: _Optional[_Union[MatrixComplex, _Mapping]] = ...,
    ) -> None: ...

class RotrayEmbResponse(_message.Message):
    __slots__ = ["xq", "xk", "xv"]
    XQ_FIELD_NUMBER: _ClassVar[int]
    XK_FIELD_NUMBER: _ClassVar[int]
    XV_FIELD_NUMBER: _ClassVar[int]
    xq: BszTensor
    xk: BszTensor
    xv: BszTensor
    def __init__(
        self,
        xq: _Optional[_Union[BszTensor, _Mapping]] = ...,
        xk: _Optional[_Union[BszTensor, _Mapping]] = ...,
        xv: _Optional[_Union[BszTensor, _Mapping]] = ...,
    ) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ["name", "data"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    data: bytes
    def __init__(
        self, name: _Optional[str] = ..., data: _Optional[bytes] = ...
    ) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ScoreRequest(_message.Message):
    __slots__ = ["bsz", "seqlen", "name", "xq", "xk", "xv", "mask"]
    BSZ_FIELD_NUMBER: _ClassVar[int]
    SEQLEN_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    XQ_FIELD_NUMBER: _ClassVar[int]
    XK_FIELD_NUMBER: _ClassVar[int]
    XV_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    bsz: int
    seqlen: int
    name: str
    xq: BszTensor
    xk: BszTensor
    xv: BszTensor
    mask: Matrix
    def __init__(
        self,
        bsz: _Optional[int] = ...,
        seqlen: _Optional[int] = ...,
        name: _Optional[str] = ...,
        xq: _Optional[_Union[BszTensor, _Mapping]] = ...,
        xk: _Optional[_Union[BszTensor, _Mapping]] = ...,
        xv: _Optional[_Union[BszTensor, _Mapping]] = ...,
        mask: _Optional[_Union[Matrix, _Mapping]] = ...,
    ) -> None: ...

class ScoreResponse(_message.Message):
    __slots__ = ["h"]
    H_FIELD_NUMBER: _ClassVar[int]
    h: Tensor
    def __init__(self, h: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class ProjRequest(_message.Message):
    __slots__ = ["name", "h"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    name: str
    h: Tensor
    def __init__(
        self, name: _Optional[str] = ..., h: _Optional[_Union[Tensor, _Mapping]] = ...
    ) -> None: ...

class ProjResponse(_message.Message):
    __slots__ = ["x"]
    X_FIELD_NUMBER: _ClassVar[int]
    x: BszTensor
    def __init__(self, x: _Optional[_Union[BszTensor, _Mapping]] = ...) -> None: ...

class FeedForwardRequest(_message.Message):
    __slots__ = ["name", "h"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    name: str
    h: Tensor
    def __init__(
        self, name: _Optional[str] = ..., h: _Optional[_Union[Tensor, _Mapping]] = ...
    ) -> None: ...

class FeedForwardResponse(_message.Message):
    __slots__ = ["h"]
    H_FIELD_NUMBER: _ClassVar[int]
    h: Tensor
    def __init__(self, h: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class OutRequest(_message.Message):
    __slots__ = ["h"]
    H_FIELD_NUMBER: _ClassVar[int]
    h: Tensor
    def __init__(self, h: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class OutResponse(_message.Message):
    __slots__ = ["output"]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    output: Matrix
    def __init__(self, output: _Optional[_Union[Matrix, _Mapping]] = ...) -> None: ...
