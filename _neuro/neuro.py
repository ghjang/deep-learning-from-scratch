import numpy as np
from typing import Self
from numpy.typing import NDArray

from common_types import LossType, OptimizerType, GradientMethod, LayerGradientDict
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

    def batch_size(self, batch_size: int) -> Self:
        self.nn.batch_size(batch_size)
        return self

    def epochs(self, epochs: int) -> Self:
        self.nn.epochs(epochs)
        return self

    def learning_rate(self, learning_rate: float) -> Self:
        self.nn.learning_rate(learning_rate)
        return self

    def loss(self, loss_type: LossType) -> Self:
        self.nn.loss(loss_type)
        return self

    def optimizer(self, optimizer_type: OptimizerType) -> Self:
        self.nn.optimizer(optimizer_type)
        return self

    def fit(
        self, x: NDArray, y: NDArray, method: GradientMethod = "backpropagation"
    ) -> list[float]:
        return self.nn.fit(x, y, method=method)

    def forward(self, x) -> NDArray:
        return self.nn.forward(x)

    def predict(self, x) -> NDArray:
        return self.nn.predict(x)

    def summary(self) -> None:
        self.nn.summary()

    def verbose(self, enable: bool = True, interval: int = 10) -> Self:
        self.nn.verbose(enable, interval)
        return self

    def compute_loss_gradients(
        self,
        x: NDArray,
        y: NDArray,
        method: GradientMethod = "numerical",
    ) -> LayerGradientDict:
        """
        '수치 미분'과 '역전파' 방식으로 계산한 '그래디언트'를 디버깅할 목적으로 작성한 메서드입니다.
        실제 학습 과정에서는 사용하지 않습니다.
        """

        if method == "numerical":
            # NOTE:
            # '수치 미분' 방식은 각 파라미터에 대해 개별적으로 미분을 계산합니다:
            # 1. 각 파라미터를 미세하게 변경(ε 만큼 증가)합니다.
            # 2. 전체 신경망에 대해 순전파(forward)를 다시 수행합니다.
            # 3. 결과 변화량으로부터 그래디언트를 추정합니다.
            # 이 방식은 구현이 간단하지만, 파라미터마다 순전파를 반복해야 하므로 계산 효율성이 낮습니다.

            # NOTE:
            # 현재의 Neuro 구현 방식에서 아직 내부적으로 레이어의 초기화 되지 않은
            # 모델 파라미터들을 초기화하기 위해서 forward 더미 호출 필요함.
            self.forward(x)

            # 손실 함수를 목적 함수로 정의
            def loss_objective(model_output: NDArray) -> float:
                return self.nn.compute_loss(model_output, y)

            # 수치 미분 방식으로 그래디언트 계산
            gradients = self.nn.compute_model_gradients(
                x=x, objective_fn=loss_objective, method=method
            )
        elif method == "backpropagation":
            # NOTE:
            # '역전파' 방식은 계산 그래프를 통한 연쇄 법칙(chain rule)을 활용합니다:
            # 1. 먼저 순전파를 한 번 수행하여 각 레이어의 입출력값을 저장합니다.
            # 2. 출력층에서 손실 함수(로스 계층)로부터 시작하여 입력층으로 역방향으로 진행하며 그래디언트를 전파합니다.
            #    - 손실 함수는 예측값과 실제값의 오차를 계산하고, 이로부터 초기 그래디언트가 결정됩니다.
            #    - 이 초기 그래디언트(dy/dŷ)는 출력층으로 전달되어 역전파가 시작됩니다.
            # 3. 각 레이어는 자신의 파라미터에 대한 그래디언트를 계산하고 이전 레이어로 그래디언트를 전달합니다.
            # 이 방식은 구현이 더 복잡하지만, 한 번의 순회로 모든 그래디언트를 계산할 수 있어 계산 효율성이 매우 높습니다.

            # 순전파 1회 수행 및 내부 각 레이어 구현에서 '중간 값(입출력 값)' 필요시 저장(캐싱)
            model_output = self.forward(x)

            # 역전파 방식으로 그래디언트 계산
            gradients = self.nn.compute_model_gradients(
                y=y, model_output=model_output, method=method
            )
        else:
            raise ValueError(f"지원하지 않는 그래디언트 계산 방법: {method}")

        return gradients
