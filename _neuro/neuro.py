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
        self.biases: NDArray | None = biases
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

    def layer(
        self, output_size: int = None, weights: NDArray = None, biases: NDArray = None
    ) -> Self:
        """
        신경망에 새 레이어를 추가합니다.

        Args:
            output_size: 레이어의 출력 크기(뉴런 수). weights가 제공된 경우 자동 계산됨
            weights: 직접 제공할 가중치 행렬. None이면 자동 초기화됨
            biases: 직접 제공할 바이어스 벡터. None이면 0으로 초기화됨

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        # 가중치가 제공된 경우 출력 크기 계산
        if weights is not None:
            calculated_output_size = weights.shape[1]
            if output_size is not None and output_size != calculated_output_size:
                raise ValueError(
                    f"제공된 가중치 행렬의 열 수({calculated_output_size})와 출력 크기({output_size})가 일치하지 않습니다."
                )
            output_size = calculated_output_size

        if output_size is None:
            raise ValueError(
                "출력 크기(output_size)나 가중치(weights)를 제공해야 합니다."
            )

        # 바이어스가 제공된 경우 크기 검증
        if biases is not None and biases.shape[0] != output_size:
            raise ValueError(
                f"제공된 바이어스 벡터의 크기({biases.shape[0]})가 출력 크기({output_size})와 일치하지 않습니다."
            )

        if len(self.layers) == 0:
            # 첫 번째 레이어 추가
            if weights is not None:
                # 직접 가중치가 제공된 경우
                if biases is None:
                    biases = np.zeros(output_size)
                self.layers.append(Layer(output_size, weights, biases))
            else:
                # weights와 biases는 forward 메서드에서 '입력 데이터 크기'에 기반해 초기화됨
                self.layers.append(Layer(output_size))
        else:
            prev_size = self.layers[-1].output_size
            if weights is None:
                # 가중치 초기화 헬퍼 메서드 활용
                weights = self._initialize_weights(prev_size, output_size)
            elif weights.shape[0] != prev_size:
                raise ValueError(
                    f"제공된 가중치 행렬의 행 수({weights.shape[0]})가 이전 레이어의 출력 크기({prev_size})와 일치하지 않습니다."
                )

            if biases is None:
                biases = np.zeros(output_size)

            self.layers.append(Layer(output_size, weights, biases))

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

    def forward(self, x: NDArray, auto_init_bias: bool = True) -> NDArray:
        """순방향 전파를 수행합니다.

        Args:
            x: 입력 데이터.
               - 1차원 배열 (features,): 단일 샘플로 처리됨
               - 2차원 배열 (batch_size, features): 배치 데이터
            auto_init_bias: True인 경우 레이어의 편향이 None인 경우 0으로 초기화합니다.

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

        layer_output = x  # NOTE: x 자체가 입력층

        # 첫 번째 레이어(첫 번째 은닉층)부터 순방향 계산 수행
        for layer in self.layers:
            # 가중치와 편향이 없으면 초기화
            if layer.weights is None:
                prev_size = layer_output.shape[1]
                # 가중치 초기화 헬퍼 메서드 활용
                layer.weights = self._initialize_weights(prev_size, layer.output_size)

            if layer.biases is None and auto_init_bias:
                layer.biases = np.zeros(layer.output_size)

            # 선형 계산: z = x @ W + b
            if layer.biases is None:
                z = layer_output @ layer.weights
            else:
                z = layer_output @ layer.weights + layer.biases

            # 활성화 함수 적용 (다음 레이어의 입력이 됨)
            layer_output = z if layer.activation is None else layer.activation(z)

        # 최종 출력(다음 레이어의 입력) 반환
        return layer_output

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

    def _format_memory_size(self, bytes_count: int) -> str:
        """바이트 수를 읽기 쉬운 메모리 크기 문자열로 변환합니다.

        Args:
            bytes_count: 바이트 단위의 메모리 크기

        Returns:
            단위가 포함된 메모리 크기 문자열 (예: "1.25 MB")
        """
        # 적절한 단위로 변환 (B, KB, MB, GB)
        units = ["B", "KB", "MB", "GB"]
        size = bytes_count
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        # 정수인 경우 소수점 없이 표시, 실수인 경우 소수점 두 자리까지 표시
        if size.is_integer():
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"

    def summary(self) -> None:
        """신경망 구조에 대한 요약 정보를 출력합니다."""
        print("신경망 모델 요약:")
        print("-" * 60)
        print(f"{'레이어':^10}{'출력 크기':^15}{'파라미터 수':^15}{'활성화 함수':^20}")
        print("-" * 60)

        total_params = 0
        total_bytes = 0  # 메모리 사용량 계산을 위한 변수 추가

        # 입력층 표시 (첫 번째 레이어의 입력 크기 기준)
        if self.layers and self.layers[0].weights is not None:
            input_size = self.layers[0].weights.shape[0]
            print(f"{'입력층':^10}{input_size:^15}{0:^15}{'없음':^20}")
        else:
            print(f"{'입력층':^10}{'알 수 없음':^15}{0:^15}{'없음':^20}")

        # 각 레이어 표시 (모두 계산에 참여하는 레이어로 취급)
        for i, layer in enumerate(self.layers):
            weights_params = 0 if layer.weights is None else layer.weights.size
            bias_params = 0 if layer.biases is None else layer.biases.size
            params = weights_params + bias_params

            # 메모리 사용량 계산 추가
            if layer.weights is not None:
                total_bytes += layer.weights.nbytes
            if layer.biases is not None:
                total_bytes += layer.biases.nbytes

            # 활성화 함수 이름 확인
            if layer.activation is None:
                act_name = "없음"
            else:
                act_name = layer.activation.__name__

            # 레이어 유형 결정 (마지막 레이어는 출력층, 나머지는 은닉층)
            layer_type = "출력층" if i == len(self.layers) - 1 else f"은닉층 {i+1}"

            total_params += params
            print(f"{layer_type:^10}{layer.output_size:^15}{params:^15}{act_name:^20}")

        # 메모리 사용량 형식화에 공통 메서드 사용
        memory_size = self._format_memory_size(total_bytes)

        print("-" * 60)
        print(f"총 파라미터 수: {total_params:,}")
        print(f"총 메모리 사용량: {memory_size}")
        print("-" * 60)

    def get_model_info(self) -> dict[str, tuple[int, str]]:
        """모델의 파라미터 개수와 메모리 사용량 정보를 반환합니다.

        한 번의 순회로 모든 정보를 수집하여 효율성을 높입니다.

        Returns:
            딕셔너리: {'parameters': (파라미터 수, 문자열 표현),
                     'memory': (바이트 수, 문자열 표현)}
        """
        total_params = 0
        total_bytes = 0

        for layer in self.layers:
            # 가중치 파라미터 처리
            if layer.weights is not None:
                total_params += layer.weights.size
                total_bytes += layer.weights.nbytes

            # 편향 파라미터 처리
            if layer.biases is not None:
                total_params += layer.biases.size
                total_bytes += layer.biases.nbytes

        # 메모리 사용량 형식화에 공통 메서드 사용
        size_str = self._format_memory_size(total_bytes)

        # 파라미터 수를 천 단위 구분자로 포맷팅
        params_str = f"{total_params:,}"

        return {
            "parameters": (total_params, params_str),
            "memory": (total_bytes, size_str),
        }

    def count_parameters(self) -> int:
        """모델의 총 학습 가능한 파라미터 개수를 반환합니다.

        Returns:
            모델의 총 파라미터 개수
        """
        return self.get_model_info()["parameters"][0]

    def memory_usage(self) -> tuple[int, str]:
        """모델 파라미터가 사용하는 총 메모리 양을 계산합니다.

        Returns:
            튜플: (바이트 단위 메모리 사용량, 단위가 포함된 문자열 표현)
        """
        return self.get_model_info()["memory"]
