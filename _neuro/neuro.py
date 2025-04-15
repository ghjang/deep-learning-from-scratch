import numpy as np
from typing import Self, override
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
    ) -> Self:
        self.nn.layer(output_size, init_weights, init_biases)
        return self

    def linear(self, matrix: NDArray) -> Self:
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
        )

        return self

    def bias_add(self, bias: NDArray | float | int) -> Self:
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
        )

        return self

    def sigmoid(self) -> Self:
        if not self.nn.layers or len(self.nn.layers) == 0:
            raise ValueError("신경망에 레이어가 없습니다.")

        prev_output_size = self.nn.layers[-1].output_size

        self.nn.layer(
            prev_output_size,
            auto_init_weights=False,
            auto_init_biases=False,
        ).activation(sigmoid)

        return self

    def forward(self, x) -> NDArray:
        return self.nn.forward(x)

    def predict(self, x) -> NDArray:
        return self.nn.predict(x)

    def summary(self) -> None:
        self.nn.summary()


if __name__ == "__main__":
    print("Neuro 클래스 테스트")
    print("-" * 50)

    # 1. 기본 모델 생성 및 구조 확인
    model = (
        Neuro.create()
        .affine(4)
        .sigmoid()
        .affine(2)
        .sigmoid()
        .linear(np.array([[0.1, 0.2], [0.3, 0.4]]))
        .bias_add(np.array([5, 10]))
    )

    # 2. 순방향 전파 테스트
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    print("\n[순방향 전파 테스트]")
    output = model.forward(X)
    print("입력 데이터 형태:", X.shape)
    print("출력 데이터 형태:", output.shape)
    print("출력 데이터:")
    print(output)

    # NOTE:
    # 'summary' 메서드 호출을 최초의 forward 전에 호출할 경우에
    # 첫번째 레이어에 아직 가중치가 초기화되지 않은 경우에 '알 수 없음'으로 표시될 수 있음.
    print("\n[모델 구조]")
    model.summary()
