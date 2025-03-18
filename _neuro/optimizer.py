import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Generic, Dict, Literal

T = TypeVar("T")


class OptimizerMixin(Generic[T]):
    """
    모델의 파라미터 최적화 기능을 제공하는 믹스인 클래스

    이 믹스인은 GradientMixin과 LossMixin의 기능을 활용하여
    손실 함수에 대한 그래디언트 계산과 모델 파라미터 업데이트 기능을 제공합니다.
    """

    def compute_loss_gradients(
        self, x: NDArray, y: NDArray, loss_type: Literal["mse", "cross_entropy"] = "mse"
    ) -> Dict[int, Dict[str, NDArray]]:
        """
        네트워크의 모든 파라미터에 대한 손실 함수 그래디언트를 계산합니다.

        이 메서드는 GradientMixin.compute_model_gradients를 활용하여
        손실 함수 계산을 위한 클로저를 만들고 그래디언트를 계산합니다.

        Args:
            x: 입력 데이터
            y: 정답 레이블
            loss_type: 손실 함수 유형

        Returns:
            각 레이어 파라미터에 대한 그래디언트 딕셔너리
        """

        # 손실 함수를 목적 함수로 정의
        def loss_objective(output: NDArray) -> float:
            return self.compute_loss(output, y, loss_type)

        # compute_model_gradients 활용 - 코드 중복 방지
        return self.compute_model_gradients(x, loss_objective)

    def update_params(
        self, gradients: Dict[int, Dict[str, NDArray]], learning_rate: float = 0.01
    ) -> None:
        """
        계산된 그래디언트를 사용하여 네트워크 파라미터를 업데이트합니다.

        Args:
            gradients: compute_gradients에서 반환된 그래디언트 정보
            learning_rate: 학습률 (기본값: 0.01)
        """
        # 각 레이어의 파라미터 업데이트
        for layer_idx, layer_grads in gradients.items():
            layer = self.layers[layer_idx]

            # 가중치 업데이트
            if "weights" in layer_grads and layer.weights is not None:
                layer.weights -= learning_rate * layer_grads["weights"]

            # 편향 업데이트
            if "biases" in layer_grads and layer.biases is not None:
                layer.biases -= learning_rate * layer_grads["biases"]

    def train_step(
        self,
        x: NDArray,
        y: NDArray,
        learning_rate: float = 0.01,
        loss_type: Literal["mse", "cross_entropy"] = "mse",
    ) -> float:
        """
        단일 학습 단계를 수행합니다.

        Args:
            x: 입력 데이터
            y: 정답 레이블
            learning_rate: 학습률
            loss_type: 손실 함수 유형

        Returns:
            현재 손실 값
        """
        # 순방향 전파
        y_pred = self.forward(x)

        # 손실 계산
        loss = self.compute_loss(y_pred, y, loss_type)

        # 그래디언트 계산 - compute_loss_gradients 사용
        gradients = self.compute_loss_gradients(x, y, loss_type)

        # 파라미터 업데이트
        self.update_params(gradients, learning_rate)

        return loss

    def fit(
        self,
        x: NDArray,
        y: NDArray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        loss_type: Literal["mse", "cross_entropy"] = "mse",
        verbose: bool = True,
    ) -> list[float]:
        """
        모델을 훈련합니다.

        Args:
            x: 입력 데이터
            y: 정답 레이블
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            loss_type: 손실 함수 유형
            verbose: 학습 진행 상황 출력 여부

        Returns:
            각 에포크별 손실값 리스트
        """
        n_samples = len(x)
        history = []

        for epoch in range(epochs):
            # 데이터 셔플
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            # 미니배치 학습
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # 단일 배치에 대한 훈련 단계 수행
                batch_loss = self.train_step(x_batch, y_batch, learning_rate, loss_type)
                epoch_loss += batch_loss * len(x_batch) / n_samples

            history.append(epoch_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"에포크 {epoch+1}/{epochs}, 손실: {epoch_loss:.6f}")

        return history
