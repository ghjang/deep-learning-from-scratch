import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Generic, Literal, Self

from layer_loss import LossLayer, MSELoss, CrossEntropyLoss

T = TypeVar("T")

type LossType = Literal["mse", "cross_entropy"]


class LossMixin(Generic[T]):
    """
    손실 함수(Loss Function) 계산을 제공하는 믹스인 클래스

    이 믹스인은 다양한 손실 함수를 구현하고 신경망 훈련에 필요한
    손실값 계산 기능을 제공합니다.

    Notes:
        SSE(Sum of Squared Error)와 MSE(Mean Squared Error)의 관계:
        -----------------------------------------------------------
        1. SSE = Σ(y_true - y_pred)²
           - 모든 오차 제곱의 합계
           - 배치 크기에 비례하여 증가함
           - 수식: SSE = Σ_i Σ_j (y_true_ij - y_pred_ij)²

        2. MSE = (1/n) x Σ(y_true - y_pred)² = SSE/n
           - 오차 제곱의 평균값
           - 배치 크기와 무관하게 일관된 크기를 유지함
           - 수식: MSE = (1/n) x Σ_i Σ_j (y_true_ij - y_pred_ij)²

        3. 배치 처리에서의 의미:
           - SSE: 배치 크기가 커지면 손실값도 비례하여 커짐
           - MSE: 배치 크기와 무관하게 일관된 스케일의 손실값 제공

        4. 그래디언트에 미치는 영향:
           - SSE 그래디언트: ∇SSE = 2(y_pred - y_true)
           - MSE 그래디언트: ∇MSE = (2/n)(y_pred - y_true) = ∇SSE/n
           - MSE 사용 시 배치 크기에 관계없이 일관된 크기의 그래디언트 계산 가능

        5. 실제 구현:
           - 딥러닝 프레임워크에서는 대부분 MSE를 사용함
           - 배치 학습 시 일관된 학습률 사용을 위해 중요
    """

    def __init__(self) -> None:
        """LossMixin 초기화"""
        super().__init__()
        self._loss_layer: LossLayer | None = None

    def loss(self, loss_type: LossType) -> Self:
        """
        사용할 손실 함수 유형을 설정합니다.

        Args:
            loss_type: 손실 함수 유형 ('mse' 또는 'cross_entropy')

        Returns:
            자기 자신 (메서드 체이닝 지원)
        """
        # 손실 함수 타입에 따라 적절한 LossLayer 객체 생성
        if loss_type == "mse":
            self._loss_layer = MSELoss()
        elif loss_type == "cross_entropy":
            self._loss_layer = CrossEntropyLoss()
        else:
            raise ValueError(f"지원하지 않는 손실 함수 유형: {loss_type}")

        return self

    @property
    def loss_layer(self) -> LossLayer | None:
        """
        현재 설정된 손실 레이어 객체를 반환합니다.
        
        이 프로퍼티를 통해 현재 모델이 사용 중인 손실 함수 레이어에 직접 접근할 수 있습니다.
        레이어 객체에 접근하여 특정 속성을 설정하거나 상태를 확인할 수 있습니다.
        
        예시:
        ```python
        # CrossEntropyLoss 객체의 axis 속성 변경
        if isinstance(model.loss_layer, CrossEntropyLoss):
            model.loss_layer.axis = 1
        ```
        
        Returns:
            설정된 손실 레이어 객체. 설정되지 않은 경우 None
        """
        return self._loss_layer

    def compute_loss(
        self,
        y_pred: NDArray,
        y_true: NDArray,
        loss_type: LossType | None = None,
        axis: int = -1,
    ) -> float:
        """
        예측값과 실제값 간의 손실을 계산합니다.

        Args:
            y_pred: 모델의 예측값
            y_true: 실제 정답값
            loss_type: 손실 함수 유형 ('mse' 또는 'cross_entropy')
                      None이면 사전 설정된 손실 레이어 사용
            axis: 분류 문제에서 클래스 차원의 축 (기본값: -1, 마지막 차원)
                - 일반적인 (batch_size, classes) 형태의 데이터에서는 axis=1
                - 다차원 출력의 경우 클래스가 있는 차원을 지정

        Returns:
            계산된 손실값

        Raises:
            RuntimeError: 손실 레이어가 설정되지 않은 경우
        """
        # loss_type이 지정된 경우 임시 LossLayer 객체 생성
        if loss_type is not None:
            if loss_type == "mse":
                temp_layer = MSELoss()
                return temp_layer.forward(y_pred, y_true)
            elif loss_type == "cross_entropy":
                temp_layer = CrossEntropyLoss(axis=axis)
                return temp_layer.forward(y_pred, y_true)
            else:
                raise ValueError(f"지원하지 않는 손실 함수 유형: {loss_type}")

        # 기본 손실 레이어 사용
        if self._loss_layer is None:
            raise RuntimeError(
                "손실 함수가 설정되지 않았습니다. loss() 메서드를 먼저 호출하세요."
            )

        # CrossEntropyLoss인 경우 axis 설정
        if isinstance(self._loss_layer, CrossEntropyLoss):
            # 현재 요청된 축과 설정된 축이 다른 경우에만 변경
            if self._loss_layer.axis != axis:
                self._loss_layer.axis = axis

        return self._loss_layer.forward(y_pred, y_true)
