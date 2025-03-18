import numpy as np
from numpy.typing import NDArray
from typing import Self
from activation import *
from model_io import SaveLoadMixin
from gradient import GradientMixin
from loss import LossMixin
from optimizer import OptimizerMixin


class Layer:
    """신경망의 레이어를 표현하는 클래스

    딕셔너리 대신 실제 클래스를 사용하여 속성 접근이 더 안전하고 명확해집니다.
    """

    def __init__(
        self,
        output_size: int,
        weights: NDArray | None = None,
        biases: NDArray | None = None,
        activation: ActivationFunction | None = None,
    ):
        self.output_size: int = output_size
        self.weights: NDArray | None = weights
        self.biases: NDArray | None = biases
        self.activation: ActivationFunction | None = activation


class NeuralNet(
    SaveLoadMixin["NeuralNet"],
    GradientMixin["NeuralNet"],
    LossMixin["NeuralNet"],
    OptimizerMixin["NeuralNet"],
):
    """신경망 모델 클래스

    SaveLoadMixin: 모델 저장/로드 기능 제공
    GradientMixin: 그래디언트 계산 기능 제공
    LossMixin: 손실 함수 계산 기능 제공
    OptimizerMixin: 모델 최적화 및 학습 기능 제공
    """

    layers: list[Layer]

    def __init__(self) -> None:
        self.layers = []

    @staticmethod
    def create() -> "NeuralNet":
        return NeuralNet()

    def layer(self, output_size: int) -> Self:
        if len(self.layers) == 0:
            # NOTE:
            # '0번째 레이어'는 '입력 레이어'이다.
            # 입력 레이어에 적용되는 'weights'와 'biases'의 '크기'는
            # 추후 실제 입력 데이터가 제공될때 결정된다.
            self.layers.append(Layer(output_size))
        else:
            prev_size = self.layers[-1].output_size
            self.layers.append(
                Layer(
                    output_size=output_size,
                    weights=np.random.randn(prev_size, output_size),
                    biases=np.random.randn(output_size),
                )
            )

        return self

    def activation(self, f: ActivationFunction) -> Self:
        self.layers[-1].activation = f
        return self

    def forward(self, x: NDArray) -> NDArray:
        """순방향 전파를 수행합니다.

        Args:
            x: 입력 데이터.
               - 1차원 배열 (features,): 단일 샘플로 처리됨
               - 2차원 배열 (batch_size, features): 배치 데이터

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

        for layer in self.layers:
            weights = layer.weights
            biases = layer.biases
            activation = layer.activation
            output_size = layer.output_size

            if weights is None:
                weights = np.random.randn(x.shape[1], output_size)
                layer.weights = weights

            if biases is None:
                biases = np.random.randn(output_size)
                layer.biases = biases

            x = x @ weights + biases

            if activation is not None:
                x = activation(x)

        return x

    def predict(self, x: NDArray) -> NDArray:
        """모델 예측을 수행합니다. 단일 샘플 또는 배치 처리를 지원합니다."""
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """주어진 입력 형상에 대한 출력 형상을 계산합니다."""
        if len(input_shape) < 2:
            raise ValueError("입력 형상은 최소 (batch_size, features)여야 합니다")

        input_batch_size = input_shape[0]
        final_output_size = self.layers[-1].output_size
        return (input_batch_size, final_output_size)


if __name__ == "__main__":
    NN = NeuralNet

    # fmt: off
    nn = NN.create()\
            .layer(50)\
            .activation(sigmoid)\
            .layer(10)\
            .activation(softmax)
    # fmt: on

    # 단건 입력 데이터 순전파 예시1
    sigle_input_data = np.random.randn(100)
    result = nn.forward(sigle_input_data)
    print("Single input data 1:\n", result)

    # 단건 입력 데이터 순전파 예시2
    sigle_input_data = np.random.randn(1, 100)
    result = nn.forward(sigle_input_data)
    print("Single input data2:\n", result)

    # 배치 입력 데이터
    batch_input_data = np.random.randn(10, 100)
    result = nn.forward(batch_input_data)
    print("Batch input data:\n", result)

    # 모델 저장 및 로드 테스트
    model = nn

    # 모델 저장 테스트
    model.save("model_test.npz", overwrite=True)  # NumPy 형식으로 저장
    model.save(
        "model_test.json", overwrite=True
    )  # JSON 형식으로 저장 (사람이 읽을 수 있는 형태)

    # 모델 로드 테스트
    loaded_model1 = NeuralNet.load("model_test.npz")
    loaded_model2 = NeuralNet.load("model_test.json")

    # 로드된 모델 테스트
    test_input = np.random.randn(1, 100)
    original_output = model.forward(test_input)
    loaded_output1 = loaded_model1.forward(test_input)
    loaded_output2 = loaded_model2.forward(test_input)

    # 원본과 로드된 모델의 출력 비교
    print("원본 모델과 로드된 모델의 출력이 일치하는지 확인:")
    print("NumPy 형식:", np.allclose(original_output, loaded_output1))
    print("JSON 형식:", np.allclose(original_output, loaded_output2))

    # 그래디언트 계산 테스트
    print("\n=== 그래디언트 계산 테스트 ===")

    # 테스트 데이터 생성
    x_test = np.random.randn(10, 100)  # 10개의 샘플, 각 100개 특성
    y_test = np.random.randn(10, 10)  # 10개의 샘플, 각 10개 출력값 (회귀 문제)

    # MSE 손실에 대한 그래디언트 계산
    gradients = nn.compute_loss_gradients(x_test, y_test, loss_type="mse")

    # 각 레이어의 그래디언트 크기 출력
    print("그래디언트 정보:")
    for layer_idx, layer_grads in gradients.items():
        if "weights" in layer_grads:
            weights_grad = layer_grads["weights"]
            print(
                f"레이어 {layer_idx} 가중치 그래디언트 - 형태: {weights_grad.shape}, 평균 크기: {np.mean(np.abs(weights_grad)):.6f}"
            )

        if "biases" in layer_grads:
            biases_grad = layer_grads["biases"]
            print(
                f"레이어 {layer_idx} 편향 그래디언트 - 형태: {biases_grad.shape}, 평균 크기: {np.mean(np.abs(biases_grad)):.6f}"
            )

    # 간단한 훈련 테스트
    print("\n=== 훈련 테스트 ===")

    # 작은 신경망 생성
    small_nn = NN.create().layer(20).activation(sigmoid).layer(1).activation(sigmoid)

    # 간단한 XOR 문제 데이터
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # 모델 훈련
    history = small_nn.fit(
        x_train, y_train, epochs=500, batch_size=4, learning_rate=0.1, verbose=True
    )

    # 예측 결과 확인
    predictions = small_nn.predict(x_train)
    print("\n예측 결과:")
    for i, x in enumerate(x_train):
        print(f"입력: {x}, 정답: {y_train[i][0]}, 예측: {predictions[i][0]:.4f}")
