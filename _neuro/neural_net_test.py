import unittest
import numpy as np

# Neuro 구현 모듈들
from activation import sigmoid
from neural_net import NeuralNet as NN
from neuro import Neuro


def compare_gradients(
    numerical_gradients, backprop_gradients, error_threshold=0.005, print_details=True
):
    """
    수치미분과 오차역전파를 통해 계산된 그래디언트를 비교하는 함수

    Args:
        numerical_gradients: 수치미분으로 계산된 그래디언트 딕셔너리
        backprop_gradients: 오차역전파로 계산된 그래디언트 딕셔너리
        error_threshold: 상대 오차 허용 임계값 (기본값: 0.005 또는 0.5%)
        print_details: 비교 세부 정보 출력 여부

    Returns:
        True if gradients match within threshold, otherwise raises AssertionError
    """
    # 1. 그래디언트 레이어 수 비교 (빈 딕셔너리 제외)
    numerical_valid_layers = {i for i, grads in numerical_gradients.items() if grads}
    backprop_valid_layers = {i for i, grads in backprop_gradients.items() if grads}

    # 두 그래디언트의 유효 레이어 인덱스 집합이 동일해야 함
    assert numerical_valid_layers == backprop_valid_layers, (
        f"그래디언트 레이어 인덱스 집합이 일치하지 않습니다.\n"
        f"수치미분: {numerical_valid_layers}, 오차역전파: {backprop_valid_layers}"
    )

    # 2. 각 레이어별 파라미터 키 집합 비교
    for layer_idx in numerical_valid_layers:
        num_param_keys = set(numerical_gradients[layer_idx].keys())
        bp_param_keys = set(backprop_gradients[layer_idx].keys())

        # 동일한 레이어 인덱스에서 파라미터 키 집합이 동일해야 함
        assert num_param_keys == bp_param_keys, (
            f"레이어 {layer_idx}의 파라미터 키 집합이 일치하지 않습니다.\n"
            f"수치미분: {num_param_keys}, 오차역전파: {bp_param_keys}"
        )

        # 3. 파라미터별 값 비교
        for param_name in num_param_keys:
            num_value = numerical_gradients[layer_idx][param_name]
            bp_value = backprop_gradients[layer_idx][param_name]

            # 방향(부호) 비교
            sign_match = np.all(np.sign(num_value) == np.sign(bp_value))
            assert (
                sign_match
            ), f"레이어 {layer_idx}의 파라미터 {param_name}의 그래디언트 방향이 일치하지 않습니다"

            # 값의 크기 비교 (상대 오차 계산)
            epsilon = 1e-10  # 0으로 나누기 방지
            relative_error = np.max(
                np.abs(bp_value - num_value)
                / np.maximum(np.maximum(np.abs(num_value), np.abs(bp_value)), epsilon)
            )

            if print_details:
                print(f"\n{param_name} 그래디언트 비교 (레이어 {layer_idx}):")
                print(f"  수치미분: {num_value}")
                print(f"  오차역전파: {bp_value}")
                print(
                    f"  상대 오차: {relative_error:.4f} (임계값: {error_threshold:.4f})"
                )

            # 상대 오차가 임계값 이하인지 확인
            assert relative_error <= error_threshold, (
                f"레이어 {layer_idx}의 {param_name} 그래디언트의 상대 오차({relative_error:.4f})가 "
                f"임계값({error_threshold:.4f})을 초과합니다"
            )

    return True


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

    def test_neuro_compute_model_gradients_compare(self):
        # fmt: off
        
        # AND 게이트 입력값
        x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        # AND 게이트 기대 출력값
        y = np.array([
            [0],
            [0],
            [0],
            [1]
        ])

        # 모델 생성 (Neuro API 사용)
        model = Neuro.create()\
                    .affine(1)\
                    .sigmoid()\
                    
        model.loss("mse")

        # fmt: on

        numerical_gradients = model.compute_loss_gradients(x, y, method="numerical")
        backprop_gradients = model.compute_loss_gradients(
            x, y, method="backpropagation"
        )

        print("\n[모델 그래디언트 비교]")
        print("Numerical Gradients:\n", numerical_gradients)
        print("Backpropagation Gradients:\n", backprop_gradients)

        # 추출한 함수를 사용하여 그래디언트 비교

        # 두 계산방식의 그래디언트 비교
        compare_gradients(numerical_gradients, backprop_gradients)

        # 모델 구조 요약 출력
        print("\n[모델 구조]")
        model.summary()

    def test_neuro_and_gate_learning(self):
        """
        Neuro API를 사용한 AND 게이트 학습 테스트
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
            [0],
            [0],
            [1]
        ])

        # 모델 생성 (Neuro API 사용)
        model = Neuro.create()\
                    .affine(1)\
                    .sigmoid()

        # 모델 학습
        model.verbose(True, interval=1000)\
                .batch_size(x.shape[0])\
                .epochs(10000)\
                .learning_rate(0.5)\
                .loss("mse")\
                .optimizer("gradient_descent")\
                .fit(x, y)

        # fmt: on

        # 훈련 데이터에 대한 예측값
        predictions = model.predict(x)
        print("\nNeuro API AND 게이트 예측 결과:\n", predictions)

        # 훈련 데이터에 대한 예측값과 실제값 비교
        self.assertTrue(np.allclose(predictions, y, atol=0.1))

        # 모델 구조 요약 출력
        print("\n[AND 게이트 - Neuro API 모델 구조]")
        model.summary()
        print("\n")

    def test_neuro_xor_gate_learning(self):
        """
        Neuro API를 사용한 XOR 게이트 학습 테스트

        XOR 게이트는 비선형 분리가 필요하므로 히든 레이어가 필요하다.
        """
        # fmt: off
        
        # 입력값
        x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        # XOR 게이트 기대 출력값
        y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])

        # 모델 생성 (Neuro API 사용)
        # XOR 문제는 은닉층이 필요하므로 2층 구조로 설계
        model = Neuro.create()\
                    .affine(2)\
                    .sigmoid()\
                    .affine(1)\
                    .sigmoid()

        # 모델 학습
        model.verbose(True, interval=1000)\
                .batch_size(x.shape[0])\
                .epochs(10000)\
                .learning_rate(0.25)\
                .loss("mse")\
                .optimizer("gradient_descent")\
                .fit(x, y)

        # fmt: on

        # 훈련 데이터에 대한 예측값
        predictions = model.predict(x)
        print("\nNeuro API XOR 게이트 예측 결과:\n", predictions)

        # 훈련 데이터에 대한 예측값과 실제값 비교
        self.assertTrue(np.allclose(predictions, y, atol=0.1))

        # 모델 구조 요약 출력
        print("\n[XOR 게이트 - Neuro API 모델 구조]")
        model.summary()


if __name__ == "__main__":
    unittest.main()
