import numpy as np
from numpy.typing import NDArray
from typing import Self
from activation import *
from model_io import SaveLoadMixin
from gradient import GradientMixin
from loss import LossMixin
from optimizer import OptimizerMixin


class Layer:
    """신경망의 레이어를 표현하는 클래스

    딕셔너리 대신 실제 클래스를 사용하여 속성 접근이 더 안전하고 명확해집니다.
    """

    def __init__(
        self,
        output_size: int,
        weights: NDArray | None = None,
        biases: NDArray | None = None,
        activation: ActivationFunction | None = None,
    ):
        self.output_size: int = output_size
        self.weights: NDArray | None = weights
        self.biases: NDArray | None = None if biases is None else biases
        self.activation: ActivationFunction | None = activation


class NeuralNet(
    SaveLoadMixin["NeuralNet"],
    GradientMixin["NeuralNet"],
    LossMixin["NeuralNet"],
    OptimizerMixin["NeuralNet"],
):
    """신경망 모델 클래스

    여러 믹스인을 상속받아 다양한 기능을 제공하는 모델 클래스입니다.

    믹스인 구성:
    - SaveLoadMixin: 모델 저장/로드 기능 제공
    - GradientMixin: 그래디언트 계산 기능 제공
    - LossMixin: 손실 함수 계산 기능 제공
    - OptimizerMixin: 모델 최적화 및 학습 기능 제공
    
    사용 예시:
    ```python
    # 모델 생성 및 구성
    model = NeuralNet.create()\
        .layer(10)\
        .activation(sigmoid)\
        .layer(1)\
        .activation(sigmoid)
        
    # 모델 학습
    model.loss("mse")\
        .optimizer("gradient_descent")\
        .learning_rate(0.1)\
        .batch_size(8)\
        .epochs(100)\
        .fit(X_train, y_train)
        
    # 예측
    predictions = model.predict(X_test)
    ```
    """

    layers: list[Layer]

    def __init__(self) -> None:
        super().__init__()
        self.layers = []

    @staticmethod
    def create() -> "NeuralNet":
        """새 신경망 모델 인스턴스를 생성합니다."""
        return NeuralNet()

    def _initialize_weights(self, input_size: int, output_size: int) -> NDArray:
        """Xavier/Glorot 초기화를 사용하여 가중치를 초기화합니다.

        Args:
            input_size: 입력 크기
            output_size: 출력 크기

        Returns:
            초기화된 가중치 행렬
        """
        weight_scale = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * weight_scale

    def layer(self, output_size: int) -> Self:
        """
        신경망에 새 레이어를 추가합니다.

        Args:
            output_size: 레이어의 출력 크기(뉴런 수)

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        if len(self.layers) == 0:
            # NOTE:
            # '0번째 레이어'는 '입력 레이어'이다.
            # 입력 레이어에 적용되는 'weights'와 'biases'의 '크기'는
            # 추후 실제 입력 데이터가 제공될때 결정된다.
            self.layers.append(Layer(output_size))
        else:
            prev_size = self.layers[-1].output_size
            # 가중치 초기화 헬퍼 메서드 활용
            weights = self._initialize_weights(prev_size, output_size)
            self.layers.append(
                Layer(
                    output_size=output_size,
                    weights=weights,
                    biases=np.zeros(output_size),  # 일반적으로 편향은 0으로 초기화
                )
            )

        return self

    def activation(self, f: ActivationFunction) -> Self:
        """
        가장 최근에 추가된 레이어에 활성화 함수를 설정합니다.

        Args:
            f: 활성화 함수

        Returns:
            자기 자신 (메서드 체이닝 지원)

        Raises:
            ValueError: 레이어가 없는 상태에서 활성화 함수를 설정하려 할 때
        """
        if not self.layers:
            raise ValueError(
                "활성화 함수를 설정하기 전에 레이어를 먼저 추가해야 합니다."
            )
        self.layers[-1].activation = f
        return self

    def forward(self, x: NDArray) -> NDArray:
        """순방향 전파를 수행합니다.

        Args:
            x: 입력 데이터.
               - 1차원 배열 (features,): 단일 샘플로 처리됨
               - 2차원 배열 (batch_size, features): 배치 데이터

        Returns:
            출력 데이터. shape=(batch_size, output_features)
        """
        # 1차원 배열을 배치 크기 1인 2차원 배열로 변환
        if x.ndim == 1:
            x = x.reshape(1, -1)  # (features,) -> (1, features)
        elif x.ndim < 2:
            raise ValueError(
                f"입력 데이터는 1차원 이상이어야 합니다. 현재 shape: {x.shape}"
            )

        # 입력층 (첫 번째 레이어)은 계산 없이 통과
        layer_input = x  # 각 레이어의 입력값

        # 두 번째 레이어부터 순방향 계산 수행
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue  # 첫 번째 레이어(입력층) 건너뛰기

            # 가중치와 편향이 없으면 초기화
            if layer.weights is None:
                prev_size = layer_input.shape[1]
                # 가중치 초기화 헬퍼 메서드 활용
                layer.weights = self._initialize_weights(prev_size, layer.output_size)

            if layer.biases is None:
                layer.biases = np.zeros(layer.output_size)

            # 선형 계산: z = x @ W + b
            z = layer_input @ layer.weights + layer.biases

            # 활성화 함수 적용 (다음 레이어의 입력이 됨)
            layer_input = z if layer.activation is None else layer.activation(z)

        # 최종 출력(다음 레이어의 입력) 반환
        return layer_input

    def predict(self, x: NDArray) -> NDArray:
        """모델 예측을 수행합니다. 단일 샘플 또는 배치 처리를 지원합니다.

        Args:
            x: 입력 데이터

        Returns:
            예측 결과
        """
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """주어진 입력 형상에 대한 출력 형상을 계산합니다.

        Args:
            input_shape: 입력 데이터의 형상 (batch_size, features, ...)

        Returns:
            출력 데이터의 형상 (batch_size, output_size)

        Raises:
            ValueError: 입력 형상이 적어도 2차원 이상이 아닌 경우
        """
        if len(input_shape) < 2:
            raise ValueError("입력 형상은 최소 (batch_size, features)여야 합니다")

        input_batch_size = input_shape[0]
        final_output_size = self.layers[-1].output_size
        return (input_batch_size, final_output_size)

    def summary(self) -> None:
        """신경망 구조에 대한 요약 정보를 출력합니다."""
        print("신경망 모델 요약:")
        print("-" * 60)
        print(f"{'레이어':^10}{'출력 크기':^15}{'파라미터 수':^15}{'활성화 함수':^20}")
        print("-" * 60)

        total_params = 0
        for i, layer in enumerate(self.layers):
            # 첫 번째 레이어는 입력층이므로 파라미터가 없음
            if i == 0:
                params = 0
                act_name = "입력층"
            else:
                weights_params = 0 if layer.weights is None else layer.weights.size
                bias_params = 0 if layer.biases is None else layer.biases.size
                params = weights_params + bias_params
                # 활성화 함수 이름 가져오기
                if layer.activation is None:
                    act_name = "없음"
                else:
                    # 함수 이름 추출
                    act_name = layer.activation.__name__

            total_params += params
            print(f"{i:^10}{layer.output_size:^15}{params:^15}{act_name:^20}")

        print("-" * 60)
        print(f"총 파라미터 수: {total_params:,}")
        print("-" * 60)
