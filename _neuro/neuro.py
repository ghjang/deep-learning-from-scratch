import numpy as np
from typing import Self
from numpy.typing import NDArray
from neural_net import NeuralNet as NN
from activation import ActivationFunction as AF, sigmoid


class Neuro:
    def __init__(self):
        self.nn = NN.create()

    @staticmethod
    def create() -> Self:
        return Neuro()

    def affine(
        self,
        output_size: int,
        init_weights: NDArray = None,
        init_biases: NDArray = None,
        name: str = None,
    ) -> Self:
        self.nn.layer(
            output_size,
            init_weights,
            init_biases,
            name="affine" if name is None else name,
        )
        return self

    def linear(self, matrix: NDArray, name: str = None) -> Self:
        if not self.nn.layers or len(self.nn.layers) == 0:
            raise ValueError("신경망에 레이어가 없습니다.")

        prev_output_size = self.nn.layers[-1].output_size

        # 행렬의 입력 차원(행)이 이전 레이어의 출력 크기와 일치하는지 확인
        if matrix.shape[0] != prev_output_size:
            raise ValueError(
                f"행렬의 행 크기({matrix.shape[0]})는 이전 레이어의 출력 크기({prev_output_size})와 일치해야 합니다."
            )

        self.nn.layer(
            init_weights=matrix,  # NOTE: 출력 크기는 'matrix.shape[1]'로 내부적으로 자동 설정됨.
            auto_init_biases=False,
            name="linear" if name is None else name,
        )

        return self

    def bias_add(self, bias: NDArray | float | int, name: str = None) -> Self:
        if not self.nn.layers or len(self.nn.layers) == 0:
            raise ValueError("신경망에 레이어가 없습니다.")

        prev_output_size = self.nn.layers[-1].output_size

        if isinstance(bias, float) or isinstance(bias, int):
            bias = np.full((prev_output_size,), bias)
        elif isinstance(bias, np.ndarray):
            if bias.shape != (prev_output_size,):
                raise ValueError(f"bias는 ({prev_output_size},) 형태여야 합니다.")
        else:
            raise ValueError("bias는 float, int 또는 np.ndarray여야 합니다.")

        self.nn.layer(
            prev_output_size,
            init_biases=bias,
            auto_init_weights=False,
            name="bias_add" if name is None else name,
        )

        return self

    def sigmoid(self, name: str = None) -> Self:
        if not self.nn.layers or len(self.nn.layers) == 0:
            raise ValueError("신경망에 레이어가 없습니다.")

        prev_output_size = self.nn.layers[-1].output_size

        self.nn.layer(
            prev_output_size,
            auto_init_weights=False,
            auto_init_biases=False,
            name="sigmoid" if name is None else name,
        ).activation(sigmoid)

        return self

    def forward(self, x) -> NDArray:
        return self.nn.forward(x)

    def predict(self, x) -> NDArray:
        return self.nn.predict(x)

    def summary(self) -> None:
        self.nn.summary()
