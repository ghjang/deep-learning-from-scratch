# 2.4 NumPy 인트로

> [목차로 돌아가기](../../README.md) | [이전: 딥러닝 라이브러리 개요](./2_3_deep_learning_libraries.md) | [다음: matplotlib 인트로](./2_5_matplotlib_intro.md)

## 2.4.0 NumPy 개요

'**NumPy(Numerical Python)**'는 파이썬의 과학 계산을 위한 핵심 라이브러리다. 효율적인 다차원 배열 객체인 `ndarray`(N-dimensional array)를 제공하며, 이를 처리하기 위한 다양한 함수들을 포함한다. 주로 다음과 같은 작업에 사용된다:

- 다차원 배열 처리
- 선형대수 연산
- 난수 생성
- 푸리에 변환
- 통계 계산

### 설치 방법

```python
# pip를 이용한 설치
pip install numpy

# conda를 이용한 설치
conda install numpy
```

## 2.4.1 배열 생성 및 기본 조작

NumPy의 핵심은 다차원 배열 객체인 `ndarray`다. 다양한 방법으로 배열을 생성할 수 있다. 이러한 함수들은 '**전역 팩토리 함수**'로, `ndarray` 객체를 생성하는 데 사용된다:

```python
import numpy as np

# 기본 배열 생성
a = np.array([1, 2, 3, 4, 5])
print(a)                      # [1 2 3 4 5]
print(type(a))                # <class 'numpy.ndarray'>

# 다차원 배열 생성
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)                      # [[1 2 3]
                              #  [4 5 6]]
print(b.shape)                # (2, 3) - 2행 3열

# 특수한 배열 생성
zeros = np.zeros((3, 4))      # 3x4 크기의 0으로 채워진 배열
ones = np.ones((2, 3, 4))     # 2x3x4 크기의 1로 채워진 배열
identity = np.eye(3)          # 3x3 단위행렬
random_array = np.random.random((2, 2))  # 0~1 사이 난수로 채워진 2x2 배열

# 범위 배열 생성
range_array = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]
linear_space = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ] - 0과 1 사이 균등하게 5개 점
```

### 배열 속성과 메서드

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 주요 속성
print(arr.ndim)       # 2 (차원 수)
print(arr.shape)      # (2, 3) (각 차원의 크기)
print(arr.size)       # 6 (요소의 총 개수)
print(arr.dtype)      # int64 (요소의 데이터 타입)

# 배열 형태 변경
reshaped = arr.reshape(3, 2)  # [[1 2]
                              #  [3 4]
                              #  [5 6]]
flattened = arr.flatten()     # [1 2 3 4 5 6]

# 배열 형태와 타입 변환
new_type = arr.astype(float)  # 배열의 데이터 타입을 float로 변환
```

### reshape 메서드와 -1 인수 활용

NumPy의 reshape 메서드는 배열 구조를 유연하게 변경할 수 있는 강력한 도구다. 특히 `-1` 인수를 사용하면 자동 차원 계산이 가능하다. 다양한 차원의 배열 형태 변환에 대한 자세한 내용은 [텐서 개념](#249-텐서와-numpy의-다차원-배열)을 참조하라.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 배열

# reshape에서 -1 사용하기
# -1은 "해당 차원의 크기를 자동으로 계산"을 의미함
auto_reshaped = arr.reshape(-1)       # [1 2 3 4 5 6] - 1차원으로 평탄화
auto_reshaped_2d = arr.reshape(3, -1) # [[1 2], [3 4], [5 6]] - 3행 ?열로 변환
```

#### -1 위치의 차원 크기 계산 방법

`-1` 인수의 차원 크기는 다음 공식으로 자동 계산된다:

```python
# 계산 공식: -1 차원 크기 = 총 원소 수 / (다른 차원들의 크기의 곱)
arr = np.arange(24)
print(f"총 원소 수: {arr.size}")  # 총 원소 수: 24

# 예시 1: reshape(6, -1)
# -1 차원 크기 = 24 / 6 = 4
print(arr.reshape(6, -1).shape)  # (6, 4)

# 예시 2: reshape(2, 3, -1)
# -1 차원 크기 = 24 / (2 × 3) = 4
print(arr.reshape(2, 3, -1).shape)  # (2, 3, 4)

# 예시 3: reshape(-1, 8)
# -1 차원 크기 = 24 / 8 = 3
print(arr.reshape(-1, 8).shape)  # (3, 8)
```

#### reshape 오류 케이스

reshape 사용 시 발생할 수 있는 일반적인 오류들:

```python
# 1. 나누어 떨어지지 않는 경우
try:
    # 원소 24개를 5행으로 나누려면 각 행이 4.8개 원소를 가져야 함 - 불가능
    result = arr.reshape(5, -1)
except ValueError as e:
    print(f"오류 발생: {e}")
    # 출력: "오류 발생: cannot reshape array of size 24 into shape (5,newaxis)"

# 2. -1을 여러 개 사용한 경우
try:
    # 두 개 이상의 차원을 -1로 지정할 수 없음
    result = arr.reshape(-1, -1)
except ValueError as e:
    print(f"오류 발생: {e}")
    # 출력: "오류 발생: can only specify one unknown dimension"

# 3. 원소 수와 맞지 않는 reshape
try:
    # 24개 원소를 2×2×10 배열로 만들려면 40개 원소가 필요
    result = arr.reshape(2, 2, 10)
except ValueError as e:
    print(f"오류 발생: {e}")
    # 출력: "오류 발생: cannot reshape array of size 24 into shape (2,2,10)"
```

#### 실전 활용 예시

```python
# 이미지 데이터 처리
image = np.random.random((100, 100, 3))  # 100x100 RGB 이미지
image_flat = image.reshape(-1, 3)        # (10000, 3) - 각 픽셀을 행으로 변환

# 배치 데이터 구성
features = np.random.random((1000, 28, 28))  # 1000개 28x28 이미지
batch_features = features.reshape(50, 20, 28, 28)  # 20개씩 50개 배치로 재구성
```

## 2.4.2 배열 인덱싱과 슬라이싱

### 파이썬 슬라이스 객체 이해하기

NumPy의 다차원 슬라이싱을 이해하기 전에, 파이썬의 내장 `slice` 타입과 대괄호(`[]`) 표기법의 내부 동작에 대해 알아보자. 기본 파이썬 슬라이싱에 대한 자세한 내용은 [1.9 시퀀스 타입과 데이터 조작](../1_python_intro/1_9_sequence_types.md#192-슬라이싱) 섹션을 참조하라:

```python
# slice 객체 직접 생성
s = slice(1, 5, None)  # 1:5와 동일
print(s)  # slice(1, 5, None)

# 리스트에 적용
a = [0, 1, 2, 3, 4, 5, 6]
print(a[s])  # [1, 2, 3, 4] - a[1:5]와 동일
```

파이썬에서 대괄호(`[]`) 표기법은 `__getitem__` 특수 메서드 호출에 대한 '문법적 편의 기능(syntactic sugar)'이다. 즉, `a[x]` 표현식은 내부적으로 `a.__getitem__(x)`로 변환된다. 여기에 콜론(`:`) 표기법을 사용한 슬라이싱을 결합하면:

```python
# 다음 세 표현식은 모두 동일하다
print(a[1:5])                        # [1, 2, 3, 4]
print(a[slice(1, 5, None)])          # [1, 2, 3, 4]
print(a.__getitem__(slice(1, 5, None)))  # [1, 2, 3, 4]
```

이처럼 파이썬의 슬라이싱 표기법 `a[시작:끝:간격]`은 내부적으로 `a.__getitem__(slice(시작, 끝, 간격))`으로 해석된다. 이 기본 메커니즘을 이해하면 이후 설명하는 NumPy의 '**다차원 슬라이싱 동작**'을 더 쉽게 이해할 수 있다.

### NumPy 배열 슬라이싱 기본

NumPy 배열은 파이썬 리스트와 유사하지만 더 강력한 인덱싱 기능을 제공한다:

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 기본 인덱싱
print(arr[0, 0])      # 1 - 첫 번째 행, 첫 번째 열
print(arr[2, 3])      # 12 - 세 번째 행, 네 번째 열

# 슬라이싱 [시작:끝:간격]
print(arr[:, 1])      # [2 6 10] - 모든 행의 두 번째 열
print(arr[1:3, :])    # [[5 6 7 8]
                      #  [9 10 11 12]] - 두 번째~세 번째 행의 모든 열
print(arr[:, ::2])    # [[1 3]
                      #  [5 7]
                      #  [9 11]] - 모든 행의 짝수 열만 선택

# 불리언 인덱싱
mask = arr > 6
print(mask)           # 각 요소가 6보다 큰지 여부를 나타내는 불리언 배열
print(arr[mask])      # [7 8 9 10 11 12] - 6보다 큰 요소들만 1차원 배열로

# 팬시 인덱싱 (정수 배열을 사용한 인덱싱)
rows = np.array([0, 2])
cols = np.array([0, 2])
print(arr[rows])                # [[1 2 3 4]
                                #  [9 10 11 12]] - 0, 2번 행 선택
print(arr[:, cols])             # [[1 3]
                                #  [5 7]
                                #  [9 11]] - 모든 행의 0, 2번 열 선택
print(arr[rows[:, np.newaxis], cols])  # [[1 3]
                                       #  [9 11]] - 특정 행과 열의 조합 선택
```

#### 다차원 슬라이싱의 내부 동작 방식

NumPy의 다차원 슬라이싱은 이러한 파이썬의 기본 메커니즘을 확장한 것이다. 다음 예제를 통해 다차원 슬라이싱과 내부 동작을 살펴보자:

```python
import numpy as np

arr_3d = np.array([
    [[0, 1], [2, 3]], 
    [[4, 5], [6, 7]], 
    [[8, 9], [10, 11]]
])  # 3x2x2 배열

# 복잡한 다차원 슬라이싱 예시
result = arr_3d[1:3, 0:2, :1]
print(result)
# [[[4]
#   [6]]
#  [[8]
#   [10]]]

# 위 슬라이싱은 내부적으로 다음과 같이 처리된다:
# arr_3d.__getitem__((slice(1, 3), slice(0, 2), slice(None, 1)))
```

여러 차원에 대한 슬라이싱 예시와 내부 해석:

```python
# 예제 1: 모든 차원에 다양한 슬라이싱 적용
subset = arr_3d[0:2, 1:, ::1]  # 첫 2개 배열, 두 번째 행부터, 모든 열
print(subset)
# [[[2 3]]
#  [[6 7]]]

# 내부적으로: arr_3d.__getitem__((slice(0, 2), slice(1, None), slice(None, None, 1)))

# 예제 2: 정수 인덱스와 슬라이스 혼합
subset = arr_3d[0, :, 1]  # 첫 번째 배열의 모든 행의 두 번째 열
print(subset)  # [1 3]

# 내부적으로: arr_3d.__getitem__((0, slice(None, None, None), 1))

# 예제 3: 음수 인덱스와 스텝 사용
subset = arr_3d[::-1, :, ::-1]  # 배열 역순, 모든 행, 열 역순
print(subset)
# [[[ 9  8]
#   [11 10]]
#  [[ 5  4]
#   [ 7  6]]
#  [[ 1  0]
#   [ 3  2]]]

# 내부적으로: arr_3d.__getitem__((slice(None, None, -1), slice(None), slice(None, None, -1)))
```

NumPy는 이러한 튜플로 구성된 여러 `slice` 객체와 정수 인덱스를 받아, 다음과 같은 단계로 처리한다:

1. 각 차원의 슬라이스/인덱스 파싱: 튜플의 각 요소를 해당 차원에 대한 슬라이싱 지시사항으로 해석
2. 스트라이드 계산: 각 차원의 슬라이스 정보를 기반으로 메모리 접근을 위한 스트라이드 값 계산
3. 메모리 뷰 생성: 원본 데이터의 메모리를 공유하는 새로운 배열 객체(뷰) 생성
4. 형상 재구성: 결과 배열의 새로운 모양(shape) 결정

파이썬의 인덱싱 해석 메커니즘을 활용함으로써, NumPy는 익숙한 구문으로 강력한 다차원 데이터 접근 방법을 제공한다.

### NumPy 다차원 슬라이싱의 구현 원리

NumPy가 파이썬의 기본 슬라이싱 문법을 다차원으로 확장할 수 있었던 이유는 파이썬의 강력한 객체 지향 기능과 특수 메서드(매직 메서드) 시스템 덕분이다:

```python
# 기본 파이썬 슬라이싱
my_list = [0, 1, 2, 3, 4]
print(my_list[1:3])  # [1, 2]

# NumPy 다차원 슬라이싱
import numpy as np
arr_2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(arr_2d[0:2, 1:3])  # [[1 2]
                          #  [4 5]]
```

#### 1. 특수 메서드를 통한 문법 확장

파이썬은 `[]` 연산자를 사용한 인덱싱과 슬라이싱을 `__getitem__` 특수 메서드로 처리한다:

```python
class SimpleDemoArray:
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, idx):
        print(f"__getitem__이 받은 인덱스: {idx}")
        return self.data[idx]
        
demo = SimpleDemoArray([10, 20, 30, 40, 50])
print(demo[1:4])  # __getitem__이 받은 인덱스: slice(1, 4, None)
```

NumPy는 이 메서드를 확장하여 쉼표로 구분된 여러 인덱스나 슬라이스를 튜플로 받아 처리한다:

```python
# arr_2d[0:2, 1:3]을 호출하면
# __getitem__((slice(0, 2, None), slice(1, 3, None)))으로 변환됨
```

#### 2. 튜플 인덱싱 및 고급 구문 분석

`a[i, j]` 구문에서 쉼표가 있으면 파이썬은 자동으로 인덱스를 튜플 `(i, j)`로 `__getitem__`에 전달한다:

```python
class MultiDimArray:
    def __getitem__(self, key):
        if isinstance(key, tuple):
            print(f"다차원 인덱싱: {key}")
            # 각 차원별로 인덱스 처리
            for dim_idx in key:
                print(f"  - 차원 인덱스: {dim_idx}")
        else:
            print(f"단일 인덱싱: {key}")
        
test = MultiDimArray()
test[1:3, 4]  # 다차원 인덱싱: (slice(1, 3, None), 4)
              #   - 차원 인덱스: slice(1, 3, None)
              #   - 차원 인덱스: 4
```

NumPy는 이러한 메커니즘을 활용해 직관적인 다차원 인덱싱 문법을 구현했다.

#### 3. 뷰와 복사 최적화

NumPy 슬라이싱의 또 다른 주요 특징은 대부분의 슬라이싱 연산이 데이터 복사 없이 '뷰(view)'를 반환한다는 점이다:

```python
# 원본 배열
arr = np.array([1, 2, 3, 4, 5])

# 슬라이스는 뷰를 반환한다
slice_view = arr[1:4]
print(slice_view)  # [2 3 4]

# 뷰를 수정하면 원본 배열도 변경된다
slice_view[0] = 20
print(arr)  # [1 20 3 4 5]
```

이는 내부적으로 배열 데이터에 대한 포인터, 형태 정보, 스트라이드(stride) 정보 등을 갖는 새로운 ndarray 객체를 생성하지만, 실제 데이터는 공유하는 방식으로 구현되었다.

#### 4. 스트라이드(stride) 기반 메모리 접근

다차원 슬라이싱의 핵심 최적화 중 하나는 스트라이드 기반 메모리 액세스다:

```python
arr = np.arange(12).reshape(3, 4)
print(arr)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 배열의 메모리 레이아웃 정보
print(f"데이터 타입 크기: {arr.itemsize} 바이트")
print(f"스트라이드: {arr.strides} 바이트")
# 스트라이드는 각 차원에서 다음 요소로 이동하기 위해 건너뛰어야 할 바이트 수
```

NumPy는 이런 방식으로 메모리에서 물리적으로 인접하지 않은 요소들도 논리적으로 인접한 것으로 표현할 수 있어, 다차원 슬라이싱이 효율적으로 동작한다.

결론적으로, NumPy의 다차원 슬라이싱은 파이썬의 기본 객체 시스템이 제공하는 유연성과 특수 메서드 확장 기능을 활용하여, 언어 레벨의 문법을 라이브러리 수준에서 자연스럽게 확장한 훌륭한 예이다.

## 2.4.3 NumPy 배열 연산

NumPy 배열은 요소별(element-wise) 연산을 지원한다:

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 기본 산술 연산
print(a + b)          # [6 8 10 12] (요소별 덧셈)
print(a - b)          # [-4 -4 -4 -4] (요소별 뺄셈)
print(a * b)          # [5 12 21 32] (요소별 곱셈)
print(a / b)          # [0.2 0.33333333 0.42857143 0.5] (요소별 나눗셈)
print(a ** 2)         # [1 4 9 16] (요소별 제곱)
print(a < b)          # [True True True True] (요소별 비교)

# 통계 연산
print(a.sum())        # 10 (모든 요소의 합)
print(a.mean())       # 2.5 (평균)
print(a.std())        # 1.118033988749895 (표준편차)
print(a.min())        # 1 (최솟값)
print(a.max())        # 4 (최댓값)
print(a.argmin())     # 0 (최솟값의 인덱스)
print(a.argmax())     # 3 (최댓값의 인덱스)

# 축(axis)을 지정한 연산
c = np.array([[1, 2], [3, 4]])
print(c.sum(axis=0))  # [4 6] (열 방향 합)
print(c.sum(axis=1))  # [3 7] (행 방향 합)
```

## 2.4.4 행렬 곱셈 연산자 `@`

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

## 2.4.5 `@` 연산자와 사용자 정의 클래스

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

## 2.4.6 브로드캐스팅

브로드캐스팅은 형태가 다른 배열 간의 연산을 가능하게 하는 NumPy의 강력한 기능이다:

```python
# 스칼라와 배열의 연산
a = np.array([1, 2, 3, 4])
print(a + 10)         # [11 12 13 14] (모든 요소에 10 더함)

# 다른 형태의 배열 간 연산
a = np.array([[1, 2, 3], [4, 5, 6]])   # 2x3 배열
b = np.array([10, 20, 30])             # 1D 배열
print(a + b)          # [[11 22 33]
                      #  [14 25 36]] (b가 각 행에 브로드캐스팅됨)

# 2D 행렬 + 열 벡터
c = np.array([[1], [2]])               # 2x1 배열 (열 벡터)
print(a + c)          # [[2 3 4]
                      #  [6 7 8]] (c가 각 열에 브로드캐스팅됨)

# 브로드캐스팅 규칙 예시
m = np.ones((3, 2))
n = np.arange(2)
print(m + n)          # [[1. 2.]
                      #  [1. 2.]
                      #  [1. 2.]]
```

## 2.4.7 유용한 NumPy 함수

NumPy는 다양한 수학 함수와 응용 함수들을 제공한다:

```python
a = np.array([-1, 2, -3, 4])

# 기본 수학 함수
print(np.abs(a))          # [1 2 3 4] (절댓값)
print(np.sqrt(np.abs(a))) # [1.         1.41421356 1.73205081 2.        ] (제곱근)
print(np.exp(a))          # [3.67879441e-01 7.38905610e+00 4.97870684e-02 5.45981500e+01] (지수)
print(np.log([1, 2, 3]))  # [0.         0.69314718 1.09861229] (자연로그)
print(np.sin(a))          # [-0.84147098  0.90929743  0.14112001 -0.7568025 ] (사인)

# 배열 조작 함수
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate([a, b]))  # [1 2 3 4 5 6] (배열 연결)

c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])
print(np.vstack([c, d]))   # [[1 2]
                          #  [3 4]
                          #  [5 6]
                          #  [7 8]] (수직 방향 연결)
print(np.hstack([c, d]))   # [[1 2 5 6]
                          #  [3 4 7 8]] (수평 방향 연결)

# 배열 분할
a = np.arange(10)
print(np.split(a, 5))     # [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7]), array([8, 9])]

# 차원 추가 및 제거
a = np.array([1, 2, 3])
print(a[:, np.newaxis])   # [[1]
                         #  [2]
                         #  [3]] (새 축 추가)
```

## 2.4.8 선형대수 연산

NumPy는 기본적인 선형대수 연산을 지원한다:

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 행렬 전치
print(a.T)               # [[1 3]
                         #  [2 4]] (행과 열 교환)

# 행렬 곱셈 (점곱)
print(np.dot(a, b))      # [[19 22]
                         #  [43 50]] (행렬 곱셈)
# 또는 '@' 연산자 사용
print(a @ b)             # [[19 22]
                         #  [43 50]] (행렬 곱셈)

# 선형 방정식 풀기
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # 방정식 Ax = b 풀이
print(x)                   # [2. 3.]

# 고유값과 고유벡터
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)         # [3.5 1.5] (고유값)
print(eigenvectors)        # [[ 0.94868332  0.31622777]
                           #  [ 0.31622777 -0.94868332]] (고유벡터)

# 행렬식과 역행렬
print(np.linalg.det(A))    # 5.0 (행렬식)
print(np.linalg.inv(A))    # [[ 0.4 -0.2]
                           #  [-0.2  0.6]] (역행렬)
```

## 2.4.9 텐서와 NumPy의 다차원 배열

'**텐서(tensor)**'는 벡터와 행렬을 일반화한 다차원 데이터 구조로, 임의의 차원 수를 가질 수 있다. 텐서를 다룰 때 [reshape 메서드](#reshape-메서드와--1-인수-활용)와 [전치 연산](#2410-전치행렬transpose과-메모리-효율성)이 자주 사용된다:

- 0차원 텐서: 스칼라 (단일 값)
- 1차원 텐서: 벡터 (값의 배열)
- 2차원 텐서: 행렬 (값의 2D 격자)
- 3차원 텐서: 큐브 (값의 3D 블록) - 예: 이미지 스택, 시공간 데이터
- 4차원 이상 텐서: 고차원 데이터 - 예: 배치 이미지 데이터, 시공간 시계열

NumPy의 `ndarray`는 수학적 텐서 개념을 컴퓨터 과학에 구현한 것으로, 임의 차원의 텐서를 효율적으로 다룰 수 있다:

```python
# NumPy로 다양한 차원의 텐서 표현
scalar = np.array(5)                  # 0차원 텐서 (스칼라)
vector = np.array([1, 2, 3])          # 1차원 텐서 (벡터)
matrix = np.array([[1, 2], [3, 4]])   # 2차원 텐서 (행렬)
cube = np.ones((2, 3, 4))             # 3차원 텐서
tesseract = np.zeros((2, 3, 4, 5))    # 4차원 텐서

print(f"스칼라 차원: {scalar.ndim}, 형태: {scalar.shape}")
print(f"벡터 차원: {vector.ndim}, 형태: {vector.shape}")
print(f"행렬 차원: {matrix.ndim}, 형태: {matrix.shape}")
print(f"3차원 텐서 차원: {cube.ndim}, 형태: {cube.shape}")
print(f"4차원 텐서 차원: {tesseract.ndim}, 형태: {tesseract.shape}")
```

텐서의 중요한 특성은 각 차원이 서로 독립적이며, 각각의 차원에서 연산을 수행하거나 조작할 수 있다는 점이다. 특히 딥러닝에서는 신경망의 가중치, 활성화 값, 그래디언트 등이 모두 텐서로 표현된다.

## 2.4.10 전치행렬(Transpose)과 메모리 효율성

NumPy에서는 행렬의 전치를 매우 간단하게 계산할 수 있다:

```python
# 행렬 전치 계산 방법
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
# [[1 2 3]
#  [4 5 6]]

# 1. .T 속성 사용 (가장 일반적인 방법)
a_t = a.T
print(a_t)
# [[1 4]
#  [2 5]
#  [3 6]]

# 2. transpose() 함수 사용
a_t2 = np.transpose(a)
print(a_t2)  # a.T와 동일 결과

# 3. 다차원 배열의 특정 축만 전치
b = np.ones((2, 3, 4))  # 2x3x4 배열
b_t = np.transpose(b, axes=(1, 0, 2))  # 3x2x4로 변환 (첫 두 차원만 교환)
print(b_t.shape)  # (3, 2, 4)
```

### N차원 배열에서의 전치(transpose) 개념

2차원 행렬에서 전치는 단순히 행과 열을 교환하는 것이지만, 3차원 이상의 텐서에서 전치는 '차원 순서의 재배열'을 의미한다. 이러한 고차원 데이터 조작에 앞서 [텐서의 개념](#249-텐서와-numpy의-다차원-배열)을 이해하는 것이 중요하다:

```python
# 3차원 배열 (2x3x4): 2개의 3행 4열 행렬 스택
tensor_3d = np.arange(24).reshape(2, 3, 4)
print("원본 형태:", tensor_3d.shape)  # (2, 3, 4)
print(tensor_3d)

# 기본 전치 (.T): 차원 순서를 완전히 뒤집음
# (2,3,4) -> (4,3,2)
transposed = tensor_3d.T
print("완전 전치 후 형태:", transposed.shape)  # (4, 3, 2)

# 특정 축만 교환하기
# 첫 번째와 두 번째 차원 교환: (2,3,4) -> (3,2,4)
trans_0_1 = np.transpose(tensor_3d, axes=(1, 0, 2))
print("축 0과 1 교환 후 형태:", trans_0_1.shape)  # (3, 2, 4)

# 시각적 이해를 위한 예시:
simple_tensor = np.array([
    [[1, 2], [3, 4], [5, 6]],       # 1층: 3x2 행렬
    [[7, 8], [9, 10], [11, 12]]     # 2층: 3x2 행렬
])  # 형태: (2, 3, 2) - 2개 층, 각 층은 3행 2열

print("원본 텐서 (2, 3, 2):")
print(simple_tensor)

# 축 0과 1을 교환: 층과 행을 교환
# (2층, 3행, 2열) -> (3층, 2행, 2열)
swapped_0_1 = np.transpose(simple_tensor, (1, 0, 2))
print("\n축 0과 1 교환 후 (3, 2, 2):")
print(swapped_0_1)

# 축 1과 2를 교환: 행과 열을 교환
# (2층, 3행, 2열) -> (2층, 2행, 3열)
swapped_1_2 = np.transpose(simple_tensor, (0, 2, 1))
print("\n축 1과 2 교환 후 (2, 2, 3):")
print(swapped_1_2)
```

#### 일반화된 전치 개념

N차원 배열에서 전치는 다음과 같이 일반화된다:

1. **완전 전치**: 모든 차원의 순서를 뒤집는다 (`.T` 속성)
   - 예: (d0, d1, d2, ..., dn) → (dn, ..., d2, d1, d0)

2. **부분 전치**: 지정된 축(차원) 간의 순서만 교환한다 (`transpose()` 함수)
   - 예: (d0, d1, d2, d3) → (d0, d2, d1, d3) (축 1과 2를 교환)

3. **임의 차원 재배열**: 원하는 순서로 차원을 완전히 재배열한다 (`transpose()` 함수)
   - 예: (d0, d1, d2, d3) → (d3, d1, d0, d2)

이러한 다차원 전치 연산은 이미지 처리, 신경망의 텐서 조작, 다차원 데이터 분석 등 다양한 영역에서 활용된다:

```python
# 4차원 예시: 배치 이미지 데이터 변환 (머신러닝에서 흔히 사용)
# (배치 크기, 높이, 너비, 채널) -> (배치 크기, 채널, 높이, 너비)
batch_images = np.random.rand(32, 128, 128, 3)  # 32장의 128x128 RGB 이미지
print("원본 형태:", batch_images.shape)  # (32, 128, 128, 3)

# PyTorch 형식으로 변환 (채널 우선)
pytorch_format = np.transpose(batch_images, (0, 3, 1, 2))
print("PyTorch 형태:", pytorch_format.shape)  # (32, 3, 128, 128)
```

#### 선형대수학적 의미

수학적으로, N차원 텐서의 전치는 텐서 곱 연산의 순서를 변경하는 것과 관련이 있다:

- 2차원 행렬 `A`에서 <code>A<sup>T</sup><sub>ij</sub> = A<sub>ji</sub></code>
- 3차원 텐서 `T`에서 <code>transpose(T, (1, 0, 2))<sub>ijk</sub> = T<sub>jik</sub></code>

고차원 텐서에서는 이렇게 인덱스의 순서를 교환하는 것이 곧 차원 간의 '역할'을 교환하는 것을 의미한다.

선형대수학에서 이런 고차원 전치 연산은 다중선형 매핑의 성질을 변형하거나, 텐서의 축약(contraction) 연산 순서를 조정하는 데 사용된다.

#### 전치행렬의 메모리 레이아웃과 성능 영향

NumPy에서 전치 연산은 데이터를 물리적으로 재배치하지 않고 **뷰(view)를 생성**하는 방식으로 동작한다. 이는 메모리 사용을 최소화하지만, 접근 패턴에 영향을 미친다:

```python
large_matrix = np.random.rand(1000, 1000)  # 큰 행렬 생성
large_transpose = large_matrix.T  # 전치행렬 (뷰)

# 메모리 레이아웃 확인
print(f"원본 행렬 스트라이드: {large_matrix.strides}")  # 예: (8000, 8) 바이트
print(f"전치 행렬 스트라이드: {large_transpose.strides}")  # 예: (8, 8000) 바이트

# 원본과 전치행렬은 동일한 메모리를 공유한다
large_matrix[0, 0] = 99
print(large_transpose[0, 0])  # 99 (동일 메모리 위치 참조)
```

**행 기반 vs 열 기반 메모리 접근 효율성:**

1. **메모리 레이아웃**: NumPy는 기본적으로 C-style 행 기반(row-major) 메모리 레이아웃을 사용한다. 즉, 같은 행의 요소들이 메모리에 연속적으로 저장된다.

2. **캐시 효율성**: 같은 행의 요소들을 순차적으로 접근할 때 캐시 효율성이 높다. 전치행렬에서는 원래 배열의 열을 접근하므로 캐시 효율성이 감소할 수 있다.

```python
import time

# 캐시 효율성 비교 실험
large_matrix = np.random.rand(5000, 5000)
large_transpose = large_matrix.T

# 행 방향 합계 계산 시간 측정
start = time.time()
row_sums = np.sum(large_matrix, axis=1)  # 행 방향 합계
row_time = time.time() - start

# 열 방향 합계 계산 시간 측정 (전치행렬의 행 합계 = 원본의 열 합계)
start = time.time()
col_sums = np.sum(large_matrix, axis=0)  # 열 방향 합계
col_time = time.time() - start

print(f"행 방향 합계 시간: {row_time:.6f}초")
print(f"열 방향 합계 시간: {col_time:.6f}초")
# 일반적으로 행 방향이 더 빠름
```

#### 메모리 레이아웃 최적화 방법

전치행렬의 성능이 중요한 경우, 메모리 레이아웃을 최적화할 수 있다:

```python
# 1. 연속적인 메모리 레이아웃으로 복사
# 뷰가 아닌 새로운 배열이 생성되므로 메모리를 더 사용하지만 접근 속도가 향상된다
contiguous_transpose = np.ascontiguousarray(large_matrix.T)
print(f"최적화된 전치행렬 스트라이드: {contiguous_transpose.strides}")  # 예: (8, 40000) 바이트

# 2. Fortran 스타일(열 우선) 배열 사용
# 이 경우 전치행렬의 접근이 더 효율적이다
f_matrix = np.asfortranarray(large_matrix)
print(f"Fortran 배열 스트라이드: {f_matrix.strides}")  # 예: (8, 5000*8) 바이트
print(f"Fortran 배열 전치 스트라이드: {f_matrix.T.strides}")  # 예: (5000*8, 8) 바이트
```

일반적인 권장 사항:

- 배열을 한 번 전치하고 여러 번 접근할 경우: `np.ascontiguousarray(A.T)`로 복사본 생성
- 전치행렬을 주로 사용할 경우: `np.asfortranarray()`로 열 우선 배열 사용
- 단순 연산이나 임시 사용 시: `.T` 속성 사용

이러한 최적화는 큰 행렬을 다루는 고성능 계산에서 중요하며, NumPy의 유연한 메모리 레이아웃 지원으로 다양한 상황에 맞게 최적화가 가능하다.

---
> [목차로 돌아가기](../../README.md) | [이전: 딥러닝 라이브러리 개요](./2_3_deep_learning_libraries.md) | [다음: matplotlib 인트로](./2_5_matplotlib_intro.md)
