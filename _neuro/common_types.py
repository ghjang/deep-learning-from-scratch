from typing import TypeAlias, Literal, Callable
from numpy.typing import NDArray

# 레이어 기본 타입을 위한 타입 별칭 정의
LayerBaseType: TypeAlias = Literal["affine", "linear", "bias_add", "passthrough"]

# 초기화 방법을 위한 타입 별칭 정의
WeightInitMethod: TypeAlias = Literal[
    "xavier", "glorot", "he", "kaiming", "normal", "uniform", "zeros"
]

# 활성화 함수 타입
type ActivationFunction = Callable[[NDArray], NDArray]

# 그래디언트 계산 방법 타입 정의
type GradientMethod = Literal["numerical", "backpropagation"]

# 최적화 알고리즘 타입 정의
type OptimizerType = Literal["gradient_descent", "momentum", "rmsprop", "adam"]

# 손실 함수 타입 정의
type LossType = Literal["mse", "cross_entropy"]
