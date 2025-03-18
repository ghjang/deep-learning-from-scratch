import numpy as np
from numpy.typing import NDArray
from typing import Self
from activation import *


class Layer(dict):
    output_size: int
    weights: NDArray | None
    biases: NDArray | None
    activation: ActivationFunction | None


class NeuralNet:
    def __init__(self) -> None:
        self.layers: list[Layer] = []

    @staticmethod
    def create() -> "NeuralNet":
        return NeuralNet()

    def layer(self, output_size: int) -> Self:
        if len(self.layers) == 0:
            # NOTE:
            # '0번째 레이어'는 '입력 레이어'이다.
            # 입력 레이어에 적용되는 'weights'와 'biases'의 '크기'는
            # 추후 실제 입력 데이터가 제공될때 결정된다.
            self.layers.append(
                {
                    "output_size": output_size,
                    "weights": None,
                    "biases": None,
                    "activation": None,
                }
            )
        else:
            self.layers.append(
                {
                    "output_size": output_size,
                    "weights": np.random.randn(
                        self.layers[-1]["output_size"], output_size
                    ),
                    "biases": np.random.randn(output_size),
                    "activation": None,
                }
            )

        return self

    def activation(self, f: ActivationFunction) -> Self:
        self.layers[-1]["activation"] = f
        return self

    def forward(self, x: NDArray) -> NDArray:
        for cur_layer in self.layers:
            if cur_layer["weights"] is None:
                cur_layer["weights"] = np.random.randn(
                    x.shape[1], cur_layer["output_size"]
                )

            if cur_layer["biases"] is None:
                cur_layer["biases"] = np.random.randn(cur_layer["output_size"])

            x = x @ cur_layer["weights"] + cur_layer["biases"]

            if cur_layer["activation"] is not None:
                x = cur_layer["activation"](x)

        return x


if __name__ == "__main__":
    input_data = np.random.randn(1, 100)

    NN = NeuralNet

    # fmt: off
    nn = NN.create()\
            .layer(50)\
            .activation(sigmoid)\
            .layer(10)\
            .activation(softmax)
    # fmt: on

    result = nn.forward(input_data)
    print(result)
