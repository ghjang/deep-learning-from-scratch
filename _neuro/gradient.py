import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Generic, Callable

# 공통 타입 임포트
from common_types import (
    NeuralParamType,
    LayerGradientDict,
    ParameterGradientDict,
    GradientMethod,
)

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

    def _compute_param_gradient(
        self,
        layer_idx: int,
        param_type: NeuralParamType,
        x: NDArray,
        objective_fn: Callable[[NDArray], float],
    ) -> NDArray:
        """
        특정 레이어와 파라미터 타입에 대한 그래디언트를 계산합니다.

        Args:
            layer_idx: 레이어 인덱스
            param_type: 파라미터 타입 ("weights" 또는 "biases")
            x: 입력 데이터
            objective_fn: 목적 함수

        Returns:
            해당 파라미터에 대한 그래디언트
        """
        model = self
        layer = model.layers[layer_idx]
        param = getattr(layer, param_type)

        if param is None:
            return None

        def obj_param(p):
            # 원본 파라미터 저장
            original_param = param.copy()
            # 새 파라미터 설정
            setattr(layer, param_type, p)
            # 순방향 계산 및 목적 함수 값 얻기
            y_pred = model.forward(x)
            result = objective_fn(y_pred)
            # 원본 파라미터 복원
            setattr(layer, param_type, original_param)
            return result

        return self.compute_function_gradients(obj_param, param)

    def compute_model_gradients(
        self,
        x: NDArray | None = None,
        y: NDArray | None = None,
        model_output: NDArray | None = None,
        objective_fn: Callable[[NDArray], float] | None = None,
        method: GradientMethod = "numerical",
    ) -> LayerGradientDict:
        """
        네트워크의 모든 파라미터에 대한 임의의 목적 함수의 그래디언트를 계산합니다.

        이 메서드는 그래디언트 계산의 일반적인 틀을 제공하며,
        특정 손실 함수에 대한 그래디언트는 OptimizerMixin에서 처리합니다.

        Args:
            x: 입력 데이터 (수치 미분 방식에서 필수)
            y: 타겟 데이터 (역전파 방식에서 필수)
                수치 미분 방식에서는 일반적으로 objective_fn 내부에서 클로저로
                캡처되므로 직접 전달하지 않아도 됩니다.
            model_output: 모델의 출력값 (역전파 방식에서 필수)
                이미 계산된 모델 출력을 제공하여 중복 계산을 방지합니다.
            objective_fn: 모델 출력을 입력으로 받아 스칼라 값을 반환하는 목적 함수 (수치 미분에서 필수)
                일반적으로 lambda output: compute_loss(output, y_target) 형태의
                클로저로 제공되며, 내부에서 y 값을 참조합니다.
            method: 그래디언트 계산 방법 ('numerical': 수치 미분, 'backpropagation': 역전파)

        참고:
            수치 미분 방식과 역전파 방식은 서로 다른 입력 매개변수를 요구합니다:
            - 수치 미분(numerical): x와 objective_fn 매개변수가 필요합니다.
              objective_fn은 보통 내부에서 y 값을 참조하는 클로저입니다.
              예시: lambda output: compute_loss(output, y_target)
            - 역전파(backpropagation): y와 model_output 매개변수가 필요합니다.

        Returns:
            각 레이어 파라미터에 대한 그래디언트 딕셔너리 (LayerGradientDict)
        """
        if method == "numerical" and x is not None and objective_fn is not None:
            # 수치 미분 방식: x 값과 objective_fn 함수가 필요함 (일반적으로 내부에서 y 값을 참조하는 클로저)
            return self._compute_numerical_gradients(x=x, objective_fn=objective_fn)
        elif method == "backpropagation" and y is not None and model_output is not None:
            # 역전파 방식: y 값과 model_output이 필요함
            return self._compute_backprop_gradients(model_output=model_output, y=y)
        else:
            raise ValueError(
                "올바른 그래디언트 계산 방법과 필요한 파라미터를 지정해야 합니다.\n"
                "- 'numerical' 방법에는 'objective_fn'이 필요합니다. (이 함수는 보통 내부적으로 y 값 참조)\n"
                "- 'backpropagation' 방법에는 'y'와 'model_output'이 직접 필요합니다."
            )

    def _compute_numerical_gradients(
        self, x: NDArray, objective_fn: Callable[[NDArray], float]
    ) -> LayerGradientDict:
        """수치 미분 방식으로 그래디언트를 계산합니다."""
        model = self
        gradients: LayerGradientDict = {}

        # 각 레이어의 파라미터에 대한 그래디언트 계산
        for i in range(len(model.layers)):
            gradients[i] = {}

            # 가중치에 대한 그래디언트 계산
            grad_weights = self._compute_param_gradient(i, "weights", x, objective_fn)
            if grad_weights is not None:
                gradients[i]["weights"] = grad_weights

            # 편향에 대한 그래디언트 계산
            grad_biases = self._compute_param_gradient(i, "biases", x, objective_fn)
            if grad_biases is not None:
                gradients[i]["biases"] = grad_biases

        return gradients

    def _compute_backprop_gradients(
        self, model_output: NDArray, y: NDArray
    ) -> LayerGradientDict:
        """역전파 방식으로 그래디언트를 계산합니다."""
        model = self
        gradients: LayerGradientDict = {}

        # NOTE:
        # 역전파 방식에서는 '손실 레이어'는 '기본 레이어'와는 별도로 처리함.
        dout = 1
        dout = model.loss_layer.backward(model_output, y, dout)

        # 역순으로 '기본 레이어' 순회
        for i in reversed(range(len(model.layers))):
            gradients[i] = {}
            layer = model.layers[i]
            dout = layer.backward(dout)

            parameter_gradients = layer.get_backprop_gradients()
            gradients[i] = parameter_gradients

        # 정렬된 그래디언트 딕셔너리 생성
        sorted_gradients: LayerGradientDict = {}
        for layer_idx in sorted(gradients.keys()):
            sorted_gradients[layer_idx] = gradients[layer_idx]

        return sorted_gradients
