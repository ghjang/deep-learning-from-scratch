import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Generic, Dict, Callable

T = TypeVar("T")


class GradientMixin(Generic[T]):
    """
    수치 미분을 이용하여 그래디언트를 계산하는 믹스인 클래스

    이 믹스인은 기본적인 수치 미분과 함수의 그래디언트를 계산하는 기능을 제공합니다.
    """

    def numerical_diff(
        self, f: Callable[[float], float], x: float, h: float = 1e-4
    ) -> float:
        """
        중앙 차분법(Central Difference Method)을 사용하여 함수 f의 x에서의 미분값을 계산합니다.

        Args:
            f: 미분할 함수
            x: 미분 지점
            h: 미분 간격 (기본값: 1e-4)

        Returns:
            x에서의 함수 f의 미분값
        """
        return (f(x + h) - f(x - h)) / (2 * h)

    def numerical_gradient(self, f: Callable[[NDArray], float], x: NDArray) -> NDArray:
        """
        수치 미분을 통해 함수 f의 x에 대한 그래디언트를 계산합니다.
        중앙 차분법(Central Difference Method)을 사용합니다.

        Args:
            f: 그래디언트를 계산할 함수
            x: 그래디언트를 계산할 지점의 입력값

        Returns:
            x에서의 함수 f의 그래디언트
        """
        h = 1e-4  # 미소 변화량
        grad = np.zeros_like(x)

        # 원본 배열의 형태와 크기 저장
        original_shape = x.shape
        x = x.reshape(-1)  # 1차원으로 평탄화
        grad = grad.reshape(-1)

        for i in range(x.size):
            # i번째 요소의 원래값 저장
            tmp_val = x[i]

            # 해당 요소에 대한 편미분 함수 정의
            def f_partial(xi):
                x[i] = xi
                result = f(x.reshape(original_shape))
                x[i] = tmp_val  # 원래 값으로 복원
                return result

            # numerical_diff를 사용하여 중앙 차분 미분 계산
            grad[i] = self.numerical_diff(f_partial, tmp_val, h)

        return grad.reshape(original_shape)

    def compute_function_gradients(
        self, f: Callable[[NDArray], float], x: NDArray
    ) -> NDArray:
        """
        임의의 함수 f에 대한 x 지점에서의 그래디언트를 계산합니다.

        Args:
            f: 그래디언트를 계산할 함수
            x: 그래디언트를 계산할 입력값

        Returns:
            x에서의 함수 f의 그래디언트
        """
        return self.numerical_gradient(f, x)

    def compute_model_gradients(
        self, x: NDArray, objective_fn: Callable[[NDArray], float]
    ) -> Dict[int, Dict[str, NDArray]]:
        """
        네트워크의 모든 파라미터에 대한 임의의 목적 함수의 그래디언트를 계산합니다.

        이 메서드는 그래디언트 계산의 일반적인 틀을 제공하며,
        특정 손실 함수에 대한 그래디언트는 OptimizerMixin에서 처리합니다.

        Args:
            x: 입력 데이터
            objective_fn: 모델 출력을 입력으로 받아 스칼라 값을 반환하는 목적 함수

        Returns:
            각 레이어 파라미터에 대한 그래디언트 딕셔너리
        """
        model = self
        gradients = {}

        # 각 레이어의 파라미터에 대한 그래디언트 계산
        for i, layer in enumerate(model.layers):
            gradients[i] = {}

            # 가중치에 대한 그래디언트 계산
            if layer.weights is not None:

                def obj_weights(weights):
                    original_weights = layer.weights.copy()
                    layer.weights = weights
                    y_pred = model.forward(x)
                    result = objective_fn(y_pred)
                    layer.weights = original_weights
                    return result

                grad_weights = self.compute_function_gradients(
                    obj_weights, layer.weights
                )
                gradients[i]["weights"] = grad_weights

            # 편향에 대한 그래디언트 계산
            if layer.biases is not None:

                def obj_biases(biases):
                    original_biases = layer.biases.copy()
                    layer.biases = biases
                    y_pred = model.forward(x)
                    result = objective_fn(y_pred)
                    layer.biases = original_biases
                    return result

                grad_biases = self.compute_function_gradients(obj_biases, layer.biases)
                gradients[i]["biases"] = grad_biases

        return gradients
