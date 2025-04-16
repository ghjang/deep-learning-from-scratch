import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias, Literal

# Neuro 구현 모듈들
from activation import ActivationFunction

# 레이어 기본 타입을 위한 타입 별칭 정의
LayerBaseType: TypeAlias = Literal["affine", "linear", "bias_add", "passthrough"]

# 초기화 방법을 위한 타입 별칭 정의
WeightInitMethod: TypeAlias = Literal[
    "xavier", "glorot", "he", "kaiming", "normal", "uniform", "zeros"
]


class Layer:
    """신경망의 레이어를 표현하는 클래스

    딕셔너리 대신 실제 클래스를 사용하여 속성 접근이 더 안전하고 명확해집니다.
    """

    def __init__(
        self,
        layer_index: int,
        output_size: int,
        name: str | None = None,
        weights: NDArray | None = None,
        biases: NDArray | None = None,
        activation: ActivationFunction | None = None,
    ):
        self.layer_index: int = layer_index
        self.output_size: int = output_size
        self.name: str | None = name

        self.weights: NDArray | None = weights
        self.biases: NDArray | None = biases
        self.activation: ActivationFunction | None = activation

        self.auto_init_weights: bool = True
        self.auto_init_biases: bool = True

    def get_base_type(self) -> LayerBaseType:
        """
        현재 레이어의 기본 타입을 설정된 속성에 따라 반환합니다.

        Returns:
            str: 레이어 기본 타입 문자열 (활성화 함수 정보 제외)
        """
        has_weights = self.weights is not None
        has_biases = self.biases is not None

        if has_weights and has_biases:
            return "affine"  # 가중치와 바이어스를 모두 가진 레이어
        elif has_weights:
            return "linear"  # 가중치만 있는 레이어 (행렬 곱셈)
        elif has_biases:
            return "bias_add"  # 바이어스만 더하는 레이어
        else:
            return "passthrough"  # 아무 변환도 하지 않는 레이어

    def get_type(self) -> str:
        """
        현재 레이어의 타입을 설정된 속성에 따라 반환합니다.

        Returns:
            str: 레이어 타입 문자열
        """
        layer_type = self.get_base_type()
        has_activation = self.activation is not None

        # 활성화 함수가 있으면 추가 정보 포함
        if has_activation:
            activation_name = self.activation.__name__
            if layer_type == "passthrough":
                layer_type = f"activation_{activation_name}"  # 활성화만 있는 경우
            else:
                layer_type = (
                    f"{layer_type}_with_{activation_name}"  # 다른 연산 + 활성화
                )

        return layer_type

    @staticmethod
    def initialize_weights(
        input_size: int, output_size: int, method: WeightInitMethod = "xavier"
    ) -> NDArray:
        """가중치 초기화를 수행하는 정적 메서드

        Args:
            input_size: 입력 크기
            output_size: 출력 크기
            method: 초기화 방법 ('xavier'/'glorot', 'he'/'kaiming', 'normal', 'uniform', 'zeros')

        Returns:
            초기화된 가중치 행렬

        Raises:
            ValueError: 지원되지 않는 초기화 방법이 지정된 경우
        """
        if method == "xavier" or method == "glorot":
            # Xavier/Glorot 초기화: 활성화 함수가 선형이거나 tanh일 때 효과적
            weight_scale = np.sqrt(2.0 / (input_size + output_size))
            return np.random.randn(input_size, output_size) * weight_scale

        elif method == "he" or method == "kaiming":
            # He/Kaiming 초기화: ReLU 계열 활성화 함수에 적합
            weight_scale = np.sqrt(2.0 / input_size)
            return np.random.randn(input_size, output_size) * weight_scale

        elif method == "normal":
            # 표준 정규 분포 초기화 (평균 0, 표준편차 0.01)
            return np.random.randn(input_size, output_size) * 0.01

        elif method == "uniform":
            # 균등 분포 초기화 (-0.05 ~ 0.05)
            return np.random.uniform(-0.05, 0.05, (input_size, output_size))

        elif method == "zeros":
            # 0으로 초기화
            return np.zeros((input_size, output_size))

        else:
            raise ValueError(
                f"지원되지 않는 초기화 방법: {method}. 'xavier', 'he', 'normal', 'uniform', 'zeros' 중 하나를 사용하세요."
            )

    def forward(
        self,
        input_data: NDArray,
        auto_init_weights: bool = True,
        auto_init_biases: bool = True,
        weight_init_method: WeightInitMethod = "xavier",
    ) -> NDArray:
        """레이어의 순방향 계산을 수행합니다.

        Args:
            input_data: 입력 데이터. shape=(batch_size, input_features)
            auto_init_weights: 가중치 자동 초기화 여부
            auto_init_biases: 바이어스 자동 초기화 여부
            weight_init_method: 가중치 초기화 방법

        Returns:
            레이어의 출력 데이터. shape=(batch_size, output_size)
        """
        # 가중치와 편향이 없으면 초기화
        if self.weights is None and self.auto_init_weights and auto_init_weights:
            prev_size = input_data.shape[1]
            self.weights = Layer.initialize_weights(
                prev_size, self.output_size, weight_init_method
            )

        if self.biases is None and self.auto_init_biases and auto_init_biases:
            self.biases = np.zeros(self.output_size)

        layer_base_type = self.get_base_type()

        match layer_base_type:
            case "affine":
                z = input_data @ self.weights + self.biases
            case "linear":
                z = input_data @ self.weights
            case "bias_add":
                z = input_data + self.biases
            case "passthrough":
                z = input_data
            case _:
                raise ValueError(f"지원되지 않는 레이어 타입: {layer_base_type}")

        # 활성화 함수 적용
        output = z if self.activation is None else self.activation(z)

        return output
