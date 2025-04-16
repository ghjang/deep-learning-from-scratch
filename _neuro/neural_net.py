import numpy as np
import warnings
from numpy.typing import NDArray
from typing import Self

# NeuralNet 클래스에서 사용하는 믹스인들
from model_io import SaveLoadMixin
from layer import Layer, WeightInitMethod
from activation import ActivationFunction
from gradient import GradientMixin
from loss import LossMixin
from optimizer import OptimizerMixin


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

    # 가중치 초기화 방법을 설정하기 위한 새로운 메서드
    def weight_initializer(self, method: WeightInitMethod = "xavier") -> Self:
        """가중치 초기화 방법을 설정합니다.

        Args:
            method: 초기화 방법
                - 'xavier'/'glorot': Xavier/Glorot 초기화 (시그모이드/tanh에 적합)
                - 'he'/'kaiming': He/Kaiming 초기화 (ReLU 계열에 적합)
                - 'normal': 표준 정규 분포 초기화
                - 'uniform': 균등 분포 초기화
                - 'zeros': 0으로 초기화

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        self._weight_init_method = method
        return self

    def layer(
        self,
        output_size: int = None,
        init_weights: NDArray = None,
        init_biases: NDArray = None,
        auto_init_weights: bool = True,
        auto_init_biases: bool = True,
        weight_init_method: WeightInitMethod = None,
        name: str = None,  # 레이어 이름 매개변수 추가
    ) -> Self:
        """신경망에 새 레이어를 추가합니다.

        Args:
            output_size: 레이어의 출력 크기(뉴런 수). weights가 제공된 경우 자동 계산됨
            init_weights: 직접 제공할 가중치 행렬. None이면 자동 초기화됨
            init_biases: 직접 제공할 바이어스 벡터. None이면 0으로 초기화됨
            auto_init_weights: 가중치 자동 초기화 여부. False인 경우 초기화하지 않음
            auto_init_biases: 바이어스 자동 초기화 여부. False인 경우 초기화하지 않음
            weight_init_method: 가중치 초기화 방법. None인 경우 모델의 기본값 사용
                - 'xavier'/'glorot': Xavier/Glorot 초기화 (시그모이드/tanh에 적합)
                - 'he'/'kaiming': He/Kaiming 초기화 (ReLU 계열에 적합)
                - 'normal': 표준 정규 분포 초기화
                - 'uniform': 균등 분포 초기화
                - 'zeros': 0으로 초기화
            name: 레이어의 이름. None인 경우 기본 이름 사용

        Returns:
            자기 자신 (메서드 체이닝 지원)

        Raises:
            ValueError: 출력 크기, 가중치 또는 바이어스 중 어느 것도 제공되지 않은 경우
            ValueError: 가중치 행렬의 형상이 이전 레이어 출력 크기와 일치하지 않는 경우
            ValueError: 바이어스 벡터의 크기가 출력 크기와 일치하지 않는 경우
        """
        # 가중치가 제공된 경우 출력 크기 계산
        if init_weights is not None:
            calculated_output_size = init_weights.shape[1]
            if output_size is not None and output_size != calculated_output_size:
                raise ValueError(
                    f"제공된 가중치 행렬의 열 수({calculated_output_size})와 출력 크기({output_size})가 일치하지 않습니다."
                )
            output_size = calculated_output_size

        if output_size is None and init_biases is not None:
            output_size = init_biases.shape[0]

        if output_size is None:
            raise ValueError(
                "'출력 크기(output_size), 가중치(weights), 바이어스(biases)' 중 하나는 반드시 제공되어야 합니다."
            )

        # 바이어스가 제공된 경우 크기 검증
        if init_biases is not None and init_biases.shape[0] != output_size:
            raise ValueError(
                f"제공된 바이어스 벡터의 크기({init_biases.shape[0]})가 출력 크기({output_size})와 일치하지 않습니다."
            )

        new_layer_index = len(self.layers) + 1

        if len(self.layers) == 0:
            # 첫 번째 레이어 추가
            if init_weights is not None:
                # 직접 가중치가 제공된 경우
                if init_biases is None and auto_init_biases:
                    init_biases = np.zeros(output_size)
                self.layers.append(
                    Layer(
                        new_layer_index,
                        output_size,
                        name,
                        init_weights,
                        init_biases,
                    )
                )
            else:
                # weights와 biases는 forward 메서드에서 '입력 데이터 크기'에 기반해 초기화됨
                self.layers.append(Layer(new_layer_index, output_size, name))
        else:
            prev_size = self.layers[-1].output_size
            if init_weights is None:
                if auto_init_weights:
                    # 가중치 초기화 방법 선택
                    method = weight_init_method or getattr(
                        self, "_weight_init_method", "xavier"
                    )
                    init_weights = Layer.initialize_weights(
                        prev_size, output_size, method
                    )
            elif init_weights.shape[0] != prev_size:
                raise ValueError(
                    f"제공된 가중치 행렬의 행 수({init_weights.shape[0]})가 이전 레이어의 출력 크기({prev_size})와 일치하지 않습니다."
                )

            if init_biases is None and auto_init_biases:
                init_biases = np.zeros(output_size)

            new_layer = Layer(
                new_layer_index, output_size, name, init_weights, init_biases
            )

            # NOTE:
            # 'forward'시에 자동으로 가중치와 편향을 초기화하지 못하게 해서
            # 단순히 '활성 함수'만 적용하는 레이어를 만들기 위한 플래그 설정임.
            new_layer.auto_init_weights = auto_init_weights
            new_layer.auto_init_biases = auto_init_biases

            self.layers.append(new_layer)

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

        last_layer = self.layers[-1]
        last_layer.activation = f

        return self

    def forward(
        self,
        x: NDArray,
        auto_init_weights: bool = True,
        auto_init_biases: bool = True,
        weight_init_method: WeightInitMethod = None,
    ) -> NDArray:
        """순방향 전파를 수행합니다.

        Args:
            x: 입력 데이터.
               - 1차원 배열 (features,): 단일 샘플로 처리됨
               - 2차원 배열 (batch_size, features): 배치 데이터
            auto_init_weights: True인 경우 레이어의 가중치가 None인 경우 자동 초기화합니다.
            auto_init_biases: True인 경우 레이어의 편향이 None인 경우 0으로 초기화합니다.

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

        # 가중치 초기화 방법 선택
        method = weight_init_method or getattr(self, "_weight_init_method", "xavier")

        # 각 레이어를 순회하면서 순방향 계산 수행
        for idx, layer in enumerate(self.layers):
            layer_output = layer.forward(
                layer_output,
                auto_init_weights,
                auto_init_biases,
                method,
            )

            layer_type = layer.get_type()
            if layer_type == "passthrough":
                warnings.warn(
                    f"경고: 레이어 {idx + 1}는 '{layer_type}' 타입으로, 입력을 그대로 출력합니다."
                )

        # 최종 출력 반환
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
        dashes = "-" * 72

        print("신경망 모델 요약:")
        print(dashes)
        print(f"{'레이어':^10}{'출력 크기':^12}{'파라미터 수':^10}{'레이어 타입':^20}")
        print(dashes)

        total_params = 0
        total_bytes = 0  # 메모리 사용량 계산을 위한 변수 추가

        # 입력층 표시 (첫 번째 레이어의 입력 크기 기준)
        if self.layers and self.layers[0].weights is not None:
            input_size = self.layers[0].weights.shape[0]
            print(f"{'입력층':^10}{input_size:^15}{0:^15}{'없음':^27}")
        else:
            print(f"{'입력층':^10}{'알 수 없음':^13}{0:^12}{'없음':^30}")

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

            # 레이어 이름 결정 (사용자 지정 이름이 있으면 사용, 없으면 기본 이름)
            if layer.name:
                layer_name = layer.name
            else:
                layer_name = "출력층" if i == len(self.layers) - 1 else f"은닉층 {i+1}"

            total_params += params

            layer_type_str = layer.get_type()
            print(
                f"{layer_name:^10}{layer.output_size:^15}{params:^15}{layer_type_str:^30}"
            )

        # 메모리 사용량 형식화에 공통 메서드 사용
        memory_size = self._format_memory_size(total_bytes)

        print(dashes)
        print(f"총 파라미터 수: {total_params:,}")
        print(f"총 메모리 사용량: {memory_size}")
        print(dashes)

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
