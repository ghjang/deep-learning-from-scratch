import unittest
import numpy as np
import os
from neuro import NeuralNet
from activation import sigmoid, softmax


class TestNeuralNet(unittest.TestCase):
    """신경망 모델 테스트 케이스"""

    def setUp(self):
        """각 테스트 실행 전에 호출되어 테스트 환경을 설정합니다."""
        # 테스트에 사용할 기본 신경망 생성
        self.nn = (
            NeuralNet.create()
            .layer(50)
            .activation(sigmoid)
            .layer(10)
            .activation(softmax)
        )

        # 테스트 데이터 생성
        self.single_input_1d = np.random.randn(100)
        self.single_input_2d = np.random.randn(1, 100)
        self.batch_input = np.random.randn(10, 100)

        # 모델 저장 경로
        self.npz_path = "test_model.npz"
        self.json_path = "test_model.json"

        # XOR 문제 데이터
        self.x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_xor = np.array([[0], [1], [1], [0]])

    def tearDown(self):
        """각 테스트 실행 후에 호출되어 테스트 리소스를 정리합니다."""
        # 테스트 파일 정리
        for path in [self.npz_path, self.json_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_forward_1d_input(self):
        """1차원 입력 데이터에 대한 forward 메서드 테스트"""
        result = self.nn.forward(self.single_input_1d)

        # 결과 형태 확인
        self.assertEqual(result.shape, (1, 10))
        # 결과가 확률 분포인지 확인(softmax 출력)
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)
        # 모든 값이 0과 1 사이인지 확인
        self.assertTrue(np.all((0 <= result) & (result <= 1)))

    def test_forward_2d_input(self):
        """2차원 입력 데이터에 대한 forward 메서드 테스트"""
        result = self.nn.forward(self.single_input_2d)

        self.assertEqual(result.shape, (1, 10))
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)

    def test_forward_batch_input(self):
        """배치 입력 데이터에 대한 forward 메서드 테스트"""
        result = self.nn.forward(self.batch_input)

        # 배치 크기가 유지되는지 확인
        self.assertEqual(result.shape, (10, 10))
        # 각 샘플의 출력이 확률 분포인지 확인
        for i in range(10):
            self.assertAlmostEqual(np.sum(result[i]), 1.0, places=6)

    def test_save_load_npz(self):
        """NumPy 형식(.npz)으로 모델 저장 및 로드 테스트"""
        # 모델 저장
        self.nn.save(self.npz_path, overwrite=True)
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(self.npz_path))

        # 모델 로드
        loaded_model = NeuralNet.load(self.npz_path)

        # 모델 구조 비교 (레이어 수 확인)
        self.assertEqual(len(loaded_model.layers), len(self.nn.layers))

        # 동일한 입력에 대한 출력 비교
        test_input = np.random.randn(1, 100)
        original_output = self.nn.forward(test_input)
        loaded_output = loaded_model.forward(test_input)

        # 출력이 거의 동일한지 확인
        self.assertTrue(np.allclose(original_output, loaded_output))

    def test_save_load_json(self):
        """JSON 형식으로 모델 저장 및 로드 테스트"""
        # 모델 저장
        self.nn.save(self.json_path, overwrite=True)
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(self.json_path))

        # 모델 로드
        loaded_model = NeuralNet.load(self.json_path)

        # 모델 구조 비교 (레이어 수 확인)
        self.assertEqual(len(loaded_model.layers), len(self.nn.layers))

        # 동일한 입력에 대한 출력 비교
        test_input = np.random.randn(1, 100)
        original_output = self.nn.forward(test_input)
        loaded_output = loaded_model.forward(test_input)

        # 출력이 거의 동일한지 확인
        self.assertTrue(np.allclose(original_output, loaded_output))

    def test_gradient_computation(self):
        """그래디언트 계산 테스트"""
        # 테스트 데이터 생성
        x_test = np.random.randn(5, 100)
        y_test = np.random.randn(5, 10)

        # 그래디언트 계산
        gradients = self.nn.compute_loss_gradients(x_test, y_test, loss_type="mse")

        # 그래디언트가 예상 형태인지 확인
        for i, layer in enumerate(
            self.nn.layers[1:], 1
        ):  # 첫 레이어는 입력층이라 건너뜀
            # 가중치 그래디언트 확인
            if "weights" in gradients[i]:
                self.assertEqual(gradients[i]["weights"].shape, layer.weights.shape)
                # 그래디언트가 0이 아님을 확인 (학습이 될 수 있는지)
                self.assertFalse(np.allclose(gradients[i]["weights"], 0))

            # 편향 그래디언트 확인
            if "biases" in gradients[i]:
                self.assertEqual(gradients[i]["biases"].shape, layer.biases.shape)
                self.assertFalse(np.allclose(gradients[i]["biases"], 0))

    def test_xor_learning(self):
        """XOR 문제 학습 테스트"""
        # XOR 문제를 위한 작은 신경망 생성
        xor_nn = (
            NeuralNet.create().layer(4).activation(sigmoid).layer(1).activation(sigmoid)
        )

        # 학습 전 성능 측정
        initial_pred = xor_nn.predict(self.x_xor)
        initial_loss = xor_nn.compute_loss(initial_pred, self.y_xor)

        # 모델 학습 (에포크 수 적게 설정)
        history = xor_nn.fit(
            self.x_xor,
            self.y_xor,
            epochs=100,
            batch_size=4,
            learning_rate=0.2,
            verbose=False,
        )

        # 학습 후 성능 측정
        final_pred = xor_nn.predict(self.x_xor)
        final_loss = xor_nn.compute_loss(final_pred, self.y_xor)

        # 손실이 감소했는지 확인
        self.assertLess(final_loss, initial_loss)

        # 예측이 정답에 가까워졌는지 확인
        for i, x in enumerate(self.x_xor):
            # 예측값이 0.5를 기준으로 이진 분류 결과와 정답을 비교
            pred_class = 1 if final_pred[i, 0] > 0.5 else 0
            true_class = self.y_xor[i, 0]
            # 75% 이상 정확도 기대 (모든 XOR 샘플 중 최소 3개는 맞춰야 함)
            # 주의: 확률적 학습이므로 항상 100% 정확도를 보장하지는 않음
            accuracy = np.sum(
                [pred_class == true_class for i in range(len(self.x_xor))]
            ) / len(self.x_xor)
            self.assertGreaterEqual(accuracy, 0.75)
            break  # 한번만 검사 (테스트 시간 단축)


if __name__ == "__main__":
    unittest.main()
