# 단순 퍼셉트론을 사용한 AND 게이트 학습 (numpy 없이)

# 입력 데이터 (AND 게이트의 진리표)
training_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]

# 가중치 초기화 (무작위 작은 값)
weights = [0.0, 0.0]
bias = 0.0

# 학습률 설정
learning_rate = 0.1
epochs = 8  # 반복 학습 횟수


# 활성화 함수 (단위 계단 함수)
def step_function(value):
    return 1 if value > 0 else 0


# 학습 과정
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for inputs, target in training_data:
        # 가중치 합 계산
        weighted_sum = weights[0] * inputs[0] + weights[1] * inputs[1] + bias

        # 활성화 함수 적용
        output = step_function(weighted_sum)

        # 가중치 업데이트 (퍼셉트론 학습 규칙)
        error = target - output
        weights[0] += learning_rate * error * inputs[0]
        weights[1] += learning_rate * error * inputs[1]
        bias += learning_rate * error

        print(
            f"입력: {inputs}, 예측: {output}, 정답: {target}, 가중치: {weights}, 바이어스: {bias}"
        )

print("\n최종 가중치:", weights)
print("최종 바이어스:", bias)


# 학습된 퍼셉트론 테스트
def perceptron_predict(x1, x2):
    weighted_sum = weights[0] * x1 + weights[1] * x2 + bias
    return step_function(weighted_sum)


print("\n=== 학습된 퍼셉트론 테스트 ===")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"입력: {x1, x2} -> 예측 결과: {perceptron_predict(x1, x2)}")
