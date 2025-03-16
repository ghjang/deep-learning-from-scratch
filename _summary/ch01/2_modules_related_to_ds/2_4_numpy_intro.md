# 2.4 numpy 인트로

> [목차로 돌아가기](../../README.md) | [이전: 딥러닝 라이브러리 개용](./2_3_deep_learning_libraries.md) | [다음: matplotlib 인트로](./2_5_matplotlib_intro.md)

## 2.4.1 행렬 곱셈 연산자 `@`

파이썬 3.5에서 도입된 행렬 곱셈 연산자 `@`는 주로 `numpy` 라이브러리의 `numpy.ndarray` 타입과 함께 사용된다. 이 연산자는 두 배열 간의 행렬 곱셈을 수행한다.

예를 들어, `numpy`를 사용하여 행렬 곱셈을 수행하는 코드는 다음과 같다:

```python
import numpy as np

# 두 개의 2x2 행렬 정의
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# 행렬 곱셈 수행
result = matrix_a @ matrix_b

print(result)
```

이 코드에서 matrix_a @ matrix_b는 matrix_a와 matrix_b의 행렬 곱셈을 수행한다. `@` 연산자는 `numpy.ndarray` 타입의 객체에 대해 행렬 곱셈을 수행하도록 정의되어 있다.

## 2.4.2 `@` 연산자와 사용자 정의 클래스

파이썬의 `@` 연산자는 `__matmul__` 메서드를 구현한 객체에 대해 동작한다. 따라서, numpy 배열뿐만 아니라, `__matmul__` 메서드를 구현한 모든 객체에 대해 `@` 연산자를 사용할 수 있다.

다음은 `__matmul__` 메서드를 구현한 사용자 정의 클래스의 예이다:

```python
class Matrix:
    def __init__(self, data):
        self.data = data

    def __matmul__(self, other):
        # 간단한 행렬 곱셈 구현 (2x2 행렬 예제)
        a, b = self.data
        c, d = other.data
        return Matrix([
            [a[0] * c[0] + a[1] * c[1], a[0] * d[0] + a[1] * d[1]],
            [b[0] * c[0] + b[1] * c[1], b[0] * d[0] + b[1] * d[1]]
        ])

    def __repr__(self):
        return f"Matrix({self.data})"

# 예제 사용
matrix1 = Matrix([[1, 2], [3, 4]])
matrix2 = Matrix([[5, 6], [7, 8]])
result = matrix1 @ matrix2
print(result)  # Matrix([[19, 22], [43, 50]])
```

이 예제에서 Matrix 클래스는 `__matmul__` 메서드를 구현하여 `@` 연산자를 사용할 수 있게 한다.

---
> [목차로 돌아가기](../../README.md) | [이전: 딥러닝 라이브러리 개용](./2_3_deep_learning_libraries.md) | [다음: matplotlib 인트로](./2_5_matplotlib_intro.md)
