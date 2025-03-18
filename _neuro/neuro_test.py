import unittest
import numpy as np
from neuro import NeuralNet as NN
from activation import sigmoid


class TestNeuralNet(unittest.TestCase):
    def test_simple_forward(self):
        # 입력값
        x = np.array([0.1, 0.2, 0.3])

        # 기대 출력값
        expected_output = np.array([[0.35]])

        # 모델 생성
        model = NN.create().layer(1)

        # 순전파 계산
        output = model.forward(x)

        # 출력값 shape 확인
        self.assertTrue(output.shape == expected_output.shape)

    def test_xor_gate_learning(self):
        # 입력값
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # 기대 출력값
        y = np.array([[0], [1], [1], [0]])

        # fmt: off
        # 모델 생성
        model = NN.create()\
                    .layer(2)\
                    .activation(sigmoid)\
                    .layer(1)\
                    .activation(sigmoid)

        # 모델 학습
        model.verbose(interval=1000)\
                .batch_size(x.shape[0])\
                .epochs(10000)\
                .learning_rate(0.25)\
                .loss("mse")\
                .optimizer("gradient_descent")\
                .fit(x, y)
        # fmt: on

        # 훈련 데이터에 대한 예측값
        predictions = model.predict(x)
        print("\nPredictions:\n", predictions)

        # 훈련 데이터에 대한 예측값과 실제값 비교
        self.assertTrue(np.allclose(predictions, y, atol=0.1))


if __name__ == "__main__":
    unittest.main()
