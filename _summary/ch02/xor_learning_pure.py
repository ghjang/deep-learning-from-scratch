import numpy as np

# NOTE: 시각화를 위한 정보를 따로 저장 없이 순수한 XOR 게이트 구현을 위한 학습 코드만을 포함함.


# 🟢 활성화 함수: 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 🔴 시그모이드 미분 (역전파용)
def sigmoid_derivative(x):
    return x * (1 - x)


# XOR 데이터셋 (입력: x1, x2 / 출력: XOR 결과)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력값
y = np.array([[0], [1], [1], [0]])  # 기대 출력 (XOR)

# 🛠 가중치 및 편향 초기화 (난수)
np.random.seed(42)  # 재현성을 위한 시드 설정
input_dim = 2  # 입력층 뉴런 개수
hidden_dim = 2  # 은닉층 뉴런 개수
output_dim = 1  # 출력층 뉴런 개수

W1 = np.random.uniform(-1, 1, (input_dim, hidden_dim))  # 입력층 → 은닉층 가중치
b1 = np.random.uniform(-1, 1, (1, hidden_dim))  # 은닉층 편향
W2 = np.random.uniform(-1, 1, (hidden_dim, output_dim))  # 은닉층 → 출력층 가중치
b2 = np.random.uniform(-1, 1, (1, output_dim))  # 출력층 편향

# 🔥 학습 설정
learning_rate = 0.5
epochs = 10000  # 학습 반복 횟수

# 🎯 학습 과정 (경사 하강법)
for epoch in range(epochs):
    # 🟢 순전파 (Forward Propagation)
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)  # 은닉층 활성화

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)  # 출력층 활성화

    # 🔴 역전파 (Backpropagation)
    output_error = y - final_output  # 출력 오차
    output_delta = output_error * sigmoid_derivative(final_output)  # 출력층 그래디언트

    hidden_error = np.dot(output_delta, W2.T)  # 은닉층 오차
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)  # 은닉층 그래디언트

    # ⚙️ 가중치 & 편향 업데이트 (경사 하강법)
    W2 += np.dot(hidden_output.T, output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += np.dot(X.T, hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # 📌 1000번마다 손실 출력
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(output_error))
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 🚀 학습 후 XOR 테스트
print("\n🎯 학습 완료! XOR 예측 결과:")
for i in range(4):
    hidden_layer = sigmoid(np.dot(X[i], W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    print(f"입력: {X[i]} → 예측: {output_layer[0][0]:.4f} (실제: {y[i][0]})")
