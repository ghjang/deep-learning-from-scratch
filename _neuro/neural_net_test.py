import unittest
import numpy as np

# Neuro 구현 모듈들
from activation import sigmoid
from neural_net import NeuralNet as NN
from neuro import Neuro


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

    def test_and_gate_learning(self):
        # fmt: off

        # 입력값
        x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        # 기대 출력값
        y = np.array([
            [0],
            [0],
            [0],
            [1]
        ])

        # 모델 생성
        model = NN.create()\
                    .layer(1)\
                    .activation(sigmoid)

        # 모델 학습
        model.verbose(interval=1000)\
                .batch_size(x.shape[0])\
                .epochs(10000)\
                .learning_rate(0.5)\
                .loss("mse")\
                .optimizer("gradient_descent")\
                .fit(x, y)

        # fmt: on

        # 훈련 데이터에 대한 예측값
        predictions = model.predict(x)
        print("\nPredictions:\n", predictions)

        # 훈련 데이터에 대한 예측값과 실제값 비교
        self.assertTrue(np.allclose(predictions, y, atol=0.1))

        # 모델 파라미터 정보 출력
        print(f"\n모델 파라미터 개수: {model.count_parameters():,}\n")
        _, memory_usage = model.memory_usage()
        print(f"모델 파라미터 메모리 사용량: {memory_usage}\n")

    def test_xor_gate_learning(self):
        """
        NOTE: 같은 '하이퍼파라미터'를 지정해서 학습을 진행하더라도
              내부에서 설정되는 '랜덤한 가중치 초기화' 등의 요소에 따라서
              학습 결과가 달라질 수 있다. 물론 경우에 따라서는 학습이
              전혀 되지 않을 수도 있다.
        """
        # fmt: off

        # 입력값
        x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        # 기대 출력값
        y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])

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

        # 모델 구조 요약 출력
        print("\n")
        model.summary()
        print("\n")


class NeuroTest(unittest.TestCase):
    def test_neuro_simple_forward(self):
        # 1. 기본 모델 생성 및 구조 확인
        model = (
            Neuro.create()
            .affine(4)
            .sigmoid()
            .affine(2, name="affine_2")
            .sigmoid()
            .linear(np.array([[0.1, 0.2], [0.3, 0.4]]))
            .bias_add(np.array([5, 10]))
        )

        # 2. 순방향 전파 테스트
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        print("\n[순방향 전파 테스트]")
        output = model.forward(X)
        print("입력 데이터 형태:", X.shape)
        print("출력 데이터 형태:", output.shape)
        print("출력 데이터:")
        print(output)

        # NOTE:
        # 'summary' 메서드 호출을 최초의 forward 전에 호출할 경우에
        # 첫번째 레이어에 아직 가중치가 초기화되지 않은 경우에 '알 수 없음'으로 표시될 수 있음.
        print("\n[모델 구조]")
        model.summary()


if __name__ == "__main__":
    unittest.main()
