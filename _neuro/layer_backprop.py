import numpy as np
from numpy.typing import NDArray
from typing import override, Callable
from abc import ABC, abstractmethod

# 공통 타입 임포트
from common_types import LayerBaseType, ActivationFunction as AF, ParameterGradientDict


class LayerBackprop(ABC):
    @staticmethod
    def create(layer_type: LayerBaseType | Callable | None) -> "LayerBackprop":
        # 문자열 타입인 경우 (LayerBaseType)
        if isinstance(layer_type, str):
            return BaseLayerBackprop(layer_type)

        # 함수인 경우 (Callable),
        elif callable(layer_type):
            af_name = getattr(layer_type, "__name__", "")

            if af_name == "sigmoid":
                return SigmoidLayerBackprop(layer_type)

            raise ValueError(
                f"지원되지 않는 활성화 함수입니다: {af_name}. 현재는 'sigmoid'만 지원합니다."
            )

        # 기타 경우
        else:
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
        """
        역전파를 수행합니다.

        Args:
            dout: 상위 레이어에서 전달된 그래디언트

        Returns:
            현재 레이어의 입력에 대한 그래디언트
        """
        pass

    def get_gradients(self) -> ParameterGradientDict:
        """
        기본 구현에서는 빈 딕셔너리를 반환합니다.
        서브클래스 레이어에서 필요시 오버라이드하여 레이어의 그래디언트를 반환하도록 구현할 수 있습니다.
        """
        return {}


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

    @override
    def get_gradients(self) -> ParameterGradientDict:
        gradients: ParameterGradientDict = {}

        if self.dW is not None:
            gradients["weights"] = self.dW

        if self.db is not None:
            gradients["biases"] = self.db

        return gradients


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
        # 현재 레이어(Sigmoid 활성화 함수 레이어)의 출력 데이터
        self.output_data = output_data

    @override
    def backward(self, dout: NDArray) -> NDArray:
        """
        Sigmoid 활성화 함수의 역전파 계산을 수행합니다.

        Args:
            dout: 상위 레이어에서 전달된 그래디언트

        Returns:
            현재 레이어의 입력에 대한 그래디언트
        """

        # Sigmoid 활성화 함수 레이어의 출력 값
        s = self.output_data

        dx = dout * (s * (1 - s))
        return dx
