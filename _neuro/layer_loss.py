import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class LossLayer(ABC):
    @abstractmethod
    def forward(self, y_pred: NDArray, y_true: NDArray) -> float:
        """손실 계산"""
        pass

    @abstractmethod
    def backward(self, y_pred: NDArray, y_true: NDArray, dout: float = 1.0) -> NDArray:
        """
        손실 함수의 그래디언트 계산
        
        Args:
            y_pred: 모델 예측값
            y_true: 실제 정답값
            dout: 상위 계층으로부터의 그래디언트 (기본값: 1.0)
                  체인룰 계산을 위한 미분값의 흐름을 나타냄
                  
        Returns:
            입력 y_pred에 대한 손실의 그래디언트
        """
        pass


class MSELoss(LossLayer):
    def forward(self, y_pred: NDArray, y_true: NDArray) -> float:
        """MSE 손실 계산"""
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_pred: NDArray, y_true: NDArray, dout: float = 1.0) -> NDArray:
        """
        MSE 손실 함수의 그래디언트 계산
        
        Args:
            y_pred: 모델의 예측값
            y_true: 실제 정답값
            dout: 상위 계층으로부터의 그래디언트 (기본값: 1.0)
            
        Returns:
            입력 y_pred에 대한 MSE의 그래디언트
        """
        batch_size = y_true.shape[0]
        # dout을 곱해서 체인 룰 적용
        return 2 * (y_pred - y_true) * dout / batch_size


class CrossEntropyLoss(LossLayer):
    def __init__(self, axis: int = -1):
        """
        교차 엔트로피 손실 레이어 초기화

        Args:
            axis: 분류 차원의 축 (기본값: -1, 마지막 차원)
        """
        self._axis = axis

    @property
    def axis(self) -> int:
        """분류 차원의 축 값을 반환합니다."""
        return self._axis

    @axis.setter
    def axis(self, value: int) -> None:
        """분류 차원의 축 값을 설정합니다."""
        self._axis = value

    def forward(self, y_pred: NDArray, y_true: NDArray) -> float:
        """교차 엔트로피 손실 계산

        Args:
            y_pred: 모델 예측값, 소프트맥스 출력값 (확률 분포)
            y_true: 실제 레이블 (원-핫 인코딩 형태)

        Returns:
            계산된 교차 엔트로피 손실값
        """
        # 수치 안정성을 위한 클리핑
        eps = 1e-10
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # 이진 분류와 다중 분류 구분
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # 이진 분류
            return -np.mean(
                y_true * np.log(y_pred_clipped)
                + (1 - y_true) * np.log(1 - y_pred_clipped)
            )
        else:
            # 다중 분류
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=self._axis))

    def backward(self, y_pred: NDArray, y_true: NDArray, dout: float = 1.0) -> NDArray:
        """
        교차 엔트로피 손실 함수의 그래디언트 계산

        교차 엔트로피 손실과 소프트맥스 활성화를 함께 사용할 경우,
        그래디언트는 y_pred - y_true 형태로 간소화됩니다.

        Args:
            y_pred: 모델 예측값 (소프트맥스 출력)
            y_true: 실제 레이블 (원-핫 인코딩)
            dout: 상위 계층으로부터의 그래디언트 (기본값: 1.0)
            
        Returns:
            입력 y_pred에 대한 교차 엔트로피 손실의 그래디언트
        """
        batch_size = y_true.shape[0]

        # 수치 안정성을 위한 클리핑
        eps = 1e-10
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # 이진 분류와 다중 분류 구분
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # 이진 분류 - 시그모이드 출력에 대한 그래디언트
            return (
                -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))
                * dout / batch_size
            )
        else:
            # 다중 분류 - 소프트맥스 출력에 대한 그래디언트
            return (y_pred_clipped - y_true) * dout / batch_size
