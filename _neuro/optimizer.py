import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Generic, Any, Literal, Self

T = TypeVar("T")

# 최적화 알고리즘 타입 정의 (더 명확한 이름 사용)
type OptimizerType = Literal["gradient_descent", "momentum", "rmsprop", "adam"]


class OptimizerMixin(Generic[T]):
    """
    모델의 파라미터 최적화 기능을 제공하는 믹스인 클래스

    이 믹스인은 GradientMixin과 LossMixin의 기능을 활용하여
    손실 함수에 대한 그래디언트 계산과 모델 파라미터 업데이트 기능을 제공합니다.

    지원하는 최적화 알고리즘:
    - gradient_descent: 경사 하강법 (배치 크기에 따라 세부 동작이 달라짐)
      - 배치 크기 = 1: 확률적 경사 하강법(SGD, Stochastic Gradient Descent)
      - 배치 크기 = 데이터 크기: 배치 경사 하강법(BGD, Batch Gradient Descent)
      - 그 외: 미니배치 경사 하강법(Mini-batch Gradient Descent)
    - 추후 다른 최적화 알고리즘 추가 예정

    주요 하이퍼파라미터:
    - learning_rate: 학습률. 각 학습 단계에서 파라미터를 얼마나 업데이트할지 결정하는 크기.
      값이 너무 크면 발산할 수 있고, 너무 작으면 학습이 느리게 진행됩니다.
    - epochs: 전체 데이터셋을 반복 학습하는 횟수.
      값이 커질수록 학습이 더 오래 진행되며, 과대적합이 발생할 수 있습니다.
    - batch_size: 한 번에 처리할 샘플의 수.
      전체 배치(batch_size=데이터 크기)는 안정적이지만 느리고,
      미니배치는 속도와 안정성 사이의 균형을 제공합니다.
    - loss_type: 손실 함수 유형('mse', 'cross_entropy' 등).
      회귀 문제에서는 'mse', 분류 문제에서는 'cross_entropy'가 일반적으로 사용됩니다.

    배치 처리의 개념 (최적화 알고리즘과 독립적):
    ------------------------------------------
    배치 처리는 전체 훈련 데이터를 더 작은 단위(배치)로 나누어 학습하는 기법입니다.
    이 개념은 특정 최적화 알고리즘에 국한되지 않고, 모든 경사 기반 최적화 방식에서 적용됩니다:

    1. 배치 처리의 주요 이점:
       - 메모리 효율성: 대용량 데이터셋을 한 번에 처리할 필요 없음
       - 계산 효율성: 부분 데이터로 더 빈번한 모델 업데이트 가능
       - 일반화 성능: 노이즈가 있는 업데이트가 지역 최적해 탈출에 도움
       - 병렬 처리: GPU 등에서 배치 단위 병렬화 가능

    2. 배치 크기 선택의 영향:
       - 작은 배치: 더 빠른 학습 반복, 더 많은 노이즈, 더 낮은 GPU 활용도
       - 큰 배치: 더 정확한 그래디언트 추정, 더 효율적인 하드웨어 활용, 지역 최적해에 갇힐 가능성
       - 최적 배치 크기는 문제, 모델, 하드웨어에 따라 다름 (일반적으로 16-256 사이)

    3. 배치 크기와 학습률의 관계:
       - 배치 크기가 크면 일반적으로 더 큰 학습률 필요
       - 배치 크기가 작으면 작은 학습률이 안정적

    4. 배치 처리 방식의 유형:
       - 배치 학습(BGD): 전체 데이터셋 사용 - 정확하지만 느림
       - 미니배치 학습: 데이터의 일부분만 사용 - 속도와 정확도의 균형
       - 온라인 학습(SGD): 한 번에 하나의 샘플만 처리 - 빠르지만 불안정

    5. 배치 처리는 모든 최적화 알고리즘(SGD, Adam, RMSProp 등)에 적용 가능한
       독립적인 개념으로, 어떤 최적화 알고리즘을 선택하든 배치 처리 전략은 별도로
       선택할 수 있습니다.

    이 믹스인에서의 배치 처리 구현:
    ------------------------------
    OptimizerMixin은 다음과 같이 배치 처리를 구현합니다:

    1. 메서드 체이닝을 통한 설정:
       - `.batch_size(n)` 메서드로 배치 크기 설정
       - `.epochs(n)` 메서드로 에포크 수 설정

    2. `fit()` 메서드에서의 데이터 배치 처리:
       - 전체 데이터셋을 무작위로 섞음(shuffle)
       - 설정된 배치 크기에 따라 데이터를 미니배치로 분할
       - 각 미니배치에 대해 순차적으로 학습 수행

    3. 구현 패턴:
       ```python
       # 배치 처리의 핵심 루프 (fit 메서드 내부)
       for i in range(0, n_samples, self._batch_size):
           x_batch = x_shuffled[i : i + self._batch_size]
           y_batch = y_shuffled[i : i + self._batch_size]
           batch_loss = self._train_step(x_batch, y_batch)
       ```

    4. 사용 예시:
       ```python
       model = NeuralNet.create()\
               .layer(10)\
               .activation(sigmoid)\
               .layer(1)\
               .activation(sigmoid)\
               .optimizer("gradient_descent")\
               .batch_size(32)  # 미니배치 크기 설정
               .epochs(100)     # 학습 에포크 수 설정
               .learning_rate(0.01)  # 학습률 설정
       
       # 배치 처리로 학습
       history = model.fit(x_train, y_train)
       ```

    경사 하강법 종류 (배치 크기에 따른 구분):
    1. 배치 경사 하강법(Batch Gradient Descent)
       - 배치 크기 = 전체 데이터셋 크기
       - 특징: 전체 데이터를 한 번에 처리하여 매우 안정적인 그래디언트 계산
       - 장점: 학습이 안정적이고, 전역 최적해로 수렴할 가능성이 높음
       - 단점: 대용량 데이터셋에서 메모리 사용량 증가, 계산 효율성 감소

    2. 확률적 경사 하강법(Stochastic Gradient Descent, SGD)
       - 배치 크기 = 1 (한 번에 하나의 샘플만 처리)
       - 특징: 매 학습 단계마다 무작위로 선택된 하나의 샘플을 사용
       - 장점: 계산 효율적, 지역 최적해를 탈출할 가능성 높음, 메모리 사용량 적음
       - 단점: 그래디언트 추정이 불안정하여 학습 과정에서 진동이 발생할 수 있음

    3. 미니배치 경사 하강법(Mini-batch Gradient Descent)
       - 배치 크기 = 중간 크기 (1 < 배치 크기 < 전체 데이터셋 크기)
       - 특징: 배치와 SGD의 장점을 결합한 방식
       - 장점: 배치의 안정성과 SGD의 효율성 사이의 균형
       - 일반적으로 2의 거듭제곱 값(예: 8, 16, 32, 64, 128)을 배치 크기로 많이 사용

    학습 과정:
        1. 매 에포크마다 전체 데이터를 무작위로 섞음 (셔플링)
        2. 지정된 batch_size만큼의 데이터로 미니배치 구성
        3. 각 미니배치에 대해 순전파, 손실 계산, 역전파, 파라미터 업데이트 수행
        4. 전체 데이터에 대해 평균 손실 계산 및 기록

    작은 데이터셋에 대한 배치 학습:
        - XOR 게이트 학습과 같이 데이터가 매우 적은 경우(예: 4개 샘플),
          전체 데이터를 한 번에 처리하는 것이 효과적일 수 있습니다.
        - 이런 경우 batch_size=데이터셋 크기(예: 4)로 설정하는 것이 좋습니다.
        - 작은 데이터셋에서는 오히려 미니배치보다 전체 배치 방식이 더 안정적인 학습을 제공합니다.
        - 작은 데이터셋에서는 epochs 값을 더 크게 설정하여 충분한 학습이 이루어지도록 해야 합니다.
    """

    # 클래스 레벨 타입 힌트 정의
    _optimizer_type: OptimizerType
    _optimizer_params: dict[str, Any]
    _optimizer_state: dict[str, Any]
    _learning_rate: float
    _epochs: int
    _batch_size: int
    _verbose: bool
    _verbose_interval: int

    def __init__(self) -> None:
        """OptimizerMixin 초기화"""
        # 기존 클래스의 __init__ 메서드 호출 (상속 체인 유지)
        super().__init__()

        # 최적화 관련 상태 초기화
        self._optimizer_type: OptimizerType = (
            "gradient_descent"  # 기본값 (더 명확한 이름)
        )
        self._optimizer_params = {}  # 최적화 파라미터 (향후 확장용)
        self._optimizer_state = {}  # 최적화 상태 저장 (향후 확장용)
        self._learning_rate = 0.01  # 기본 학습률
        self._epochs = 100  # 기본 에포크 수
        self._batch_size = 32  # 기본 배치 크기
        self._verbose = True  # 기본 출력 여부
        self._verbose_interval = 100  # 기본 출력 간격

    def optimizer(
        self, optimizer_type: OptimizerType = "gradient_descent", **kwargs
    ) -> Self:
        """
        사용할 최적화 알고리즘을 설정합니다.

        Args:
            optimizer_type: 최적화 알고리즘 유형
                          - 'gradient_descent': 배치 크기에 따른 경사 하강법
                          - 추후 다른 알고리즘 지원 예정
            **kwargs: 최적화 알고리즘에 필요한 추가 파라미터

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        # 최적화 방법 설정
        self._optimizer_type = optimizer_type
        self._optimizer_params = kwargs

        # 최적화 알고리즘별 상태 변수 초기화
        self._optimizer_state = {}  # 상태 초기화

        # 현재는 gradient_descent만 구현되어 있으므로 다른 타입이 들어오면 경고
        if optimizer_type != "gradient_descent":
            print(
                f"경고: 현재 '{optimizer_type}' 최적화 알고리즘은 구현되어 있지 않습니다. 'gradient_descent'로 대체됩니다."
            )

        return self

    def learning_rate(self, rate: float) -> Self:
        """
        학습률을 설정합니다.

        Args:
            rate: 학습률 값

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        self._learning_rate = rate
        return self

    def epochs(self, n: int) -> Self:
        """
        학습 에포크 수를 설정합니다.

        Args:
            n: 학습 반복 횟수

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        self._epochs = n
        return self

    def batch_size(self, n: int) -> Self:
        """
        미니배치 크기를 설정합니다.

        Args:
            n: 배치당 샘플 수

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        self._batch_size = n
        return self

    def verbose(self, enabled: bool = True, interval: int = 10) -> Self:
        """
        학습 과정 출력 여부와 출력 간격을 설정합니다.

        Args:
            enabled: 학습 진행 상황 출력 여부 (기본값: True)
            interval: 학습 상태를 출력할 에포크 간격 (기본값: 10)
                     예: interval=10이면 10, 20, 30... 에포크마다 상태 출력

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        self._verbose = enabled
        self._verbose_interval = interval
        return self

    def compute_loss_gradients(
        self, x: NDArray, y: NDArray
    ) -> dict[int, dict[str, NDArray]]:
        """
        네트워크의 모든 파라미터에 대한 손실 함수 그래디언트를 계산합니다.

        이 메서드는 GradientMixin.compute_model_gradients를 활용하여
        손실 함수 계산을 위한 클로저를 만들고 그래디언트를 계산합니다.

        Args:
            x: 입력 데이터
            y: 정답 레이블

        Returns:
            각 레이어 파라미터에 대한 그래디언트 딕셔너리
        """

        # 손실 함수를 목적 함수로 정의
        def loss_objective(output: NDArray) -> float:
            return self.compute_loss(output, y)

        return self.compute_model_gradients(x, loss_objective)

    def _update_params(self, gradients: dict[int, dict[str, NDArray]]) -> None:
        """
        계산된 그래디언트를 사용하여 네트워크 파라미터를 업데이트합니다.

        Args:
            gradients: compute_gradients에서 반환된 그래디언트 정보
        """
        # 현재는 경사 하강법만 지원
        self._update_gradient_descent(gradients, self._learning_rate)

    def _update_gradient_descent(
        self, gradients: dict[int, dict[str, NDArray]], learning_rate: float
    ) -> None:
        """SGD 알고리즘으로 파라미터 업데이트"""
        for layer_idx, layer_grads in gradients.items():
            layer = self.layers[layer_idx]

            # 가중치 업데이트
            if "weights" in layer_grads and layer.weights is not None:
                layer.weights -= learning_rate * layer_grads["weights"]

            # 편향 업데이트
            if "biases" in layer_grads and layer.biases is not None:
                layer.biases -= learning_rate * layer_grads["biases"]

    def _train_step(self, x: NDArray, y: NDArray) -> float:
        """
        단일 학습 단계를 수행합니다.

        Args:
            x: 입력 데이터
            y: 정답 레이블

        Returns:
            현재 손실 값
        """
        # 순방향 전파
        y_pred = self.forward(x)

        # 손실 계산 - LossMixin의 _loss_type 사용
        loss = self.compute_loss(y_pred, y)

        # 그래디언트 계산 - compute_loss_gradients 사용
        gradients = self.compute_loss_gradients(x, y)

        # 파라미터 업데이트
        self._update_params(gradients)

        return loss

    def fit(
        self,
        x: NDArray,
        y: NDArray,
    ) -> list[float]:
        """
        모델을 훈련합니다. 배치 크기에 따른 경사 하강법을 사용합니다.

        Args:
            x: 전체 훈련 데이터 입력값. 형태: (샘플 수, 특성 수)
            y: 전체 훈련 데이터의 정답값. 형태: (샘플 수, 출력 크기)

        Returns:
            각 에포크별 손실값 리스트
        """
        n_samples = len(x)
        history = []

        # 현재 사용 중인 경사 하강법 유형 확인
        if self._batch_size == 1:
            descent_type = "확률적 경사 하강법(SGD)"
        elif self._batch_size >= n_samples:
            descent_type = "배치 경사 하강법(BGD)"
        else:
            descent_type = "미니배치 경사 하강법(Mini-batch GD)"

        if self._verbose:
            print(
                f"학습 시작: {descent_type}, 배치 크기={min(self._batch_size, n_samples)}, 에포크={self._epochs}"
            )

        for epoch in range(self._epochs):
            # 데이터 셔플
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            # 미니배치 학습
            for i in range(0, n_samples, self._batch_size):
                x_batch = x_shuffled[i : i + self._batch_size]
                y_batch = y_shuffled[i : i + self._batch_size]

                # 단일 배치에 대한 훈련 단계 수행
                batch_loss = self._train_step(x_batch, y_batch)
                epoch_loss += batch_loss * len(x_batch) / n_samples

            history.append(epoch_loss)

            if self._verbose and (
                epoch % self._verbose_interval == 0 or epoch == self._epochs - 1
            ):
                print(f"에포크 {epoch+1}/{self._epochs}, 손실: {epoch_loss:.6f}")

        return history
