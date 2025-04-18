import numpy as np
from numpy.typing import NDArray

# 공통 타입 임포트
from common_types import ActivationFunction


def identity(x: NDArray) -> NDArray:
    """항등 함수: 입력을 그대로 반환합니다."""
    return x


def step(x: NDArray) -> NDArray:
    """계단 함수: 입력이 0보다 크면 1, 그렇지 않으면 0을 반환합니다."""
    return 1.0 * (x > 0)


def relu(x: NDArray) -> NDArray:
    """ReLU 함수: 입력이 0보다 크면 그대로, 그렇지 않으면 0을 반환합니다."""
    return np.maximum(0, x)


def leaky_relu(x: NDArray, alpha: float = 0.01) -> NDArray:
    """Leaky ReLU 함수: 입력이 0보다 크면 그대로, 아니면 alpha를 곱합니다.

    Args:
        x: 입력 배열
        alpha: 음수 입력에 대한 기울기 계수 (기본값: 0.01)

    Returns:
        Leaky ReLU 활성화 적용된 배열
    """
    return np.where(x > 0, x, alpha * x)


def elu(x: NDArray, alpha: float = 1.0) -> NDArray:
    """ELU(Exponential Linear Unit) 활성화 함수

    Args:
        x: 입력 배열
        alpha: 음수 영역의 스케일 파라미터 (기본값: 1.0)

    Returns:
        ELU 활성화 적용된 배열
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def sigmoid(x: NDArray) -> NDArray:
    """시그모이드 함수: 입력을 0과 1 사이의 값으로 변환합니다.

    큰 음수 입력값에 대한 오버플로우 방지를 위한 클리핑 적용
    """
    # 수치 안정성을 위해 입력값 클리핑
    x_safe = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_safe))


def tanh(x: NDArray) -> NDArray:
    """쌍곡선 탄젠트 함수: 입력을 -1과 1 사이의 값으로 변환합니다."""
    return np.tanh(x)


def softmax(x: NDArray, axis: int | None = -1) -> NDArray:
    """소프트맥스 함수: 입력을 확률 분포로 변환합니다.

    Parameters:
        x: 입력 배열
        axis: 확률 분포를 계산할 축 (기본값: -1, 마지막 차원)
            - 일반적인 2D 입력 (batch_size, features)에서는 axis=1
            - 일반적인 다차원 입력에서는 axis=-1 (마지막 차원)

    이 함수는 오버플로우 방지를 위해 지수 계산 전에 최대값을 뺍니다.
    """
    # 수치 안정성을 위해 지수 함수 적용 전 최대값을 빼기
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
