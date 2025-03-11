import numpy as np
import matplotlib.pyplot as plt
import json

# XOR 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# 신경망 구조 정의
input_size = 2
hidden_size = 2  # 은닉층 뉴런 2개
output_size = 1

# 가중치 및 편향 초기화
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# 학습 설정
learning_rate = 0.1
epochs = 10000
log_data = []  # 학습 과정 저장 리스트


# 활성화 함수 (시그모이드)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 학습 루프
for epoch in range(epochs):
    # Forward
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # 손실 계산 (MSE)
    error = Y - final_output
    loss = np.mean(error**2)

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)

    # 가중치 & 편향 업데이트
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0) * learning_rate
    W1 += np.dot(X.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0) * learning_rate

    # 매 100 에포크마다 가중치 기록
    if epoch % 100 == 0:
        line1_slope = -W1[0, 0] / W1[1, 0]
        line1_intercept = -b1[0] / W1[1, 0]
        line2_slope = -W1[0, 1] / W1[1, 1]
        line2_intercept = -b1[1] / W1[1, 1]
        log_data.append(
            {
                "epoch": epoch,
                "line1": [line1_slope, line1_intercept],
                "line2": [line2_slope, line2_intercept],
            }
        )

# 학습 로그 저장
with open("xor_training_log.json", "w") as f:
    json.dump(log_data, f)

print("학습 완료! 로그 파일 저장됨: xor_training_log.json")
