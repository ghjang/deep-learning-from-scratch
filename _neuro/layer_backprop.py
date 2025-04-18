import numpy as np
from numpy.typing import NDArray
from typing import override
from abc import ABC, abstractmethod

from layer import LayerBaseType
from activation import ActivationFunction as AF


class LayerBackprop(ABC):
    @staticmethod
    def create(layer_type: LayerBaseType | AF | None) -> "LayerBackprop":
        match layer_type:
            case str():  # LayerBaseType
                return BaseLayerBackprop(layer_type)

            case AF():
                # 활성화 함수 이름
                af_name = layer_type.__name__

                if af_name == "sigmoid":
                    return SigmoidLayerBackprop(layer_type)

                raise ValueError(
                    "LayerBackprop.create() method does not accept activation function."
                )

            case _:
                return NullLayerBackprop()

    @abstractmethod
    def forward_layer_data(
        self,
        weights: NDArray,
        biases: NDArray,
        input_data: NDArray,
        output_data: NDArray,
    ) -> None:
        pass

    @abstractmethod
    def backward(self, dout: NDArray) -> NDArray:
        pass


class NullLayerBackprop(LayerBackprop):
    @override
    def forward_layer_data(
        self,
        weights: NDArray,
        biases: NDArray,
        input_data: NDArray,
        output_data: NDArray,
    ) -> None:
        pass

    @override
    def backward(self, dout: NDArray) -> NDArray:
        return dout


class BaseLayerBackprop(LayerBackprop):
    layer_type: LayerBaseType
    weights: NDArray | None
    biases: NDArray | None
    input_data: NDArray | None
    output_data: NDArray | None
    dW: NDArray | None
    db: NDArray | None

    def __init__(self, layer_type: LayerBaseType):
        self.layer_type = layer_type
        self.weights = None
        self.biases = None
        self.input_data = None
        self.output_data = None
        self.dW = None
        self.db = None

    @override
    def forward_layer_data(
        self,
        weights: NDArray,
        biases: NDArray,
        input_data: NDArray,
        output_data: NDArray,
    ) -> None:
        match self.layer_type:
            case "affine":
                self.weights = weights  # 현재 레이어의 가중치
                self.input_data = input_data  # 현재 레이어의 입력 데이터

            case "linear":
                pass

            case "bias_add":
                pass

            case _:
                raise ValueError(f"지원되지 않는 레이어 타입: {self.layer_type}")

    @override
    def backward(self, dout: NDArray) -> NDArray:
        match self.layer_type:
            case "affine":
                x = self.input_data
                W = self.weights

                self.dW = x.T @ dout
                self.db = np.sum(dout, axis=0)

                dx = dout @ W.T
                return dx

            case "linear":
                raise NotImplementedError(
                    "Linear layer backpropagation not implemented."
                )

            case "bias_add":
                raise NotImplementedError(
                    "Bias add layer backpropagation not implemented."
                )

            case _:
                raise ValueError(f"지원되지 않는 레이어 타입: {self.layer_type}")


class SigmoidLayerBackprop(LayerBackprop):
    layer_type: LayerBaseType | AF
    output_data: NDArray | None

    def __init__(self, layer_type: LayerBaseType | AF):
        self.layer_type = layer_type
        self.output_data = None

    @override
    def forward_layer_data(
        self,
        weights: NDArray,
        biases: NDArray,
        input_data: NDArray,
        output_data: NDArray,
    ) -> None:
        # 현재 레이어의 출력 데이터
        self.output_data = output_data

    @override
    def backward(self, dout: NDArray) -> NDArray:
        y = self.output_data

        dx = dout * (y * (1 - y))
        return dx
