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

# 파라미터 타입 정의
type NeuralParamType = Literal["weights", "biases"]

# 단일 레이어의 파라미터별 그래디언트를 저장하는 딕셔너리 타입 정의
# - 키(str): 파라미터 타입 ("weights" 또는 "biases")
# - 값(NDArray): 해당 파라미터에 대한 그래디언트 배열
type ParameterGradientDict = dict[str, NDArray]

# 신경망의 모든 레이어의 그래디언트 정보를 저장하는 딕셔너리 타입 정의
# - 키(int): 레이어 인덱스
# - 값(ParameterGradientDict): 해당 레이어의 파라미터별 그래디언트 정보
type LayerGradientDict = dict[int, ParameterGradientDict]
