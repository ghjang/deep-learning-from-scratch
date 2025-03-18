# 2.4 NumPy 인트로

> [목차로 돌아가기](../../README.md) | [이전: 딥러닝 라이브러리 개요](./2_3_deep_learning_libraries.md) | [다음: NumPy 심화](./2_5_numpy_deep_dive.md)

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

NumPy의 reshape 메서드는 배열 구조를 유연하게 변경할 수 있는 강력한 도구다. 특히 `-1` 인수를 사용하면 자동 차원 계산이 가능하다. 다양한 차원의 배열 형태 변환에 대한 자세한 내용은 [텐서 개념](./2_5_numpy_deep_dive.md#252-텐서와-numpy의-다차원-배열)을 참조하라.

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

## 2.4.3 배열 마스킹(Boolean Masking)

마스킹은 불리언(Boolean) 배열을 사용하여 특정 조건을 만족하는 요소만 선택하는 강력한 인덱싱 방식이다. 불리언 마스킹을 이용하면 복잡한 조건에 기반한 데이터 필터링을 효율적으로 수행할 수 있다:

```python
import numpy as np

# 기본 마스킹 예제
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5  # 5보다 큰 요소에 대해서만 True 값을 갖는 불리언 배열 생성
print(mask)     # [False False False False False  True  True  True  True  True]
filtered = arr[mask]  # 마스크를 사용하여 5보다 큰 요소만 선택
print(filtered)  # [ 6  7  8  9 10]

# 한 줄로 간단히 표현
filtered = arr[arr > 5]  # 위와 동일한 결과
print(filtered)  # [ 6  7  8  9 10]
```

### 다차원 배열 마스킹

2차원 이상의 배열에서도 마스킹을 적용할 수 있다:

```python
# 2차원 배열 마스킹
arr_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# 5보다 큰 요소 선택
mask_2d = arr_2d > 5
print(mask_2d)
# [[False False False False]
#  [False  True  True  True]
#  [ True  True  True  True]]

# 마스크를 적용하면 1차원 배열로 반환됨
filtered_2d = arr_2d[mask_2d]
print(filtered_2d)  # [ 6  7  8  9 10 11 12]

# 행 단위 마스킹: 행의 모든 요소가 특정 값보다 큰 행만 선택
row_mask = np.all(arr_2d > 3, axis=1)  # 각 행의 모든 요소가 3보다 큰지 검사
print(row_mask)  # [False False  True]
print(arr_2d[row_mask])  # [[ 9 10 11 12]]

# 열 단위 마스킹도 가능
col_mask = np.any(arr_2d > 8, axis=0)  # 각 열에 8보다 큰 값이 하나라도 있는지 검사
print(col_mask)  # [ True  True  True  True]
print(arr_2d[:, col_mask])  # 모든 열이 조건을 만족하므로 원본 그대로 반환
```

#### np.all과 np.any 함수

배열 요소에 대한 조건을 집계하는 `np.all`과 `np.any` 함수는 배열 마스킹에서 매우 유용하다:

- **`np.all(condition, axis=None)`**:
  - 모든 요소가 True일 때만 True 반환
  - `axis` 매개변수가 지정되면, 해당 축을 따라 모든 값이 True인지 검사
  - 예: `np.all([True, False, True])` → `False`

- **`np.any(condition, axis=None)`**:
  - 하나 이상의 요소가 True이면 True 반환
  - `axis` 매개변수가 지정되면, 해당 축을 따라 하나라도 True 값이 있는지 검사
  - 예: `np.any([True, False, True])` → `True`

이 함수들은 복잡한 조건 필터링을 간결하게 표현할 수 있게 해준다:

```python
# 예제: 배열에서 양수만 포함하는 행 찾기
data = np.array([
    [1, 2, 3],    # 모두 양수
    [-1, 0, 2],   # 음수 포함
    [0, 0, 0]     # 모두 0
])

# 행 단위로 모든 요소가 양수인지 검사
positive_rows = np.all(data > 0, axis=1)
print(positive_rows)  # [ True False False]
print(data[positive_rows])  # [[1 2 3]]

# 행 단위로 하나라도 양수가 있는지 검사
has_positive = np.any(data > 0, axis=1)
print(has_positive)  # [ True  True False]
print(data[has_positive])  # [[1 2 3], [-1 0 2]]
```

이러한 함수들은 데이터 필터링, 조건 검사, 품질 관리 등 다양한 데이터 분석 작업에서 널리 활용된다.

### 복합 조건 마스킹

여러 조건을 조합하여 복잡한 마스킹을 만들 수 있다:

```python
# 논리 연산자를 사용한 복합 조건
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# AND 조건: 3보다 크고 8보다 작은 요소
mask_and = (arr > 3) & (arr < 8)
print(arr[mask_and])  # [4 5 6 7]

# OR 조건: 3보다 작거나 8보다 큰 요소
mask_or = (arr < 3) | (arr > 8)
print(arr[mask_or])  # [ 1  2  9 10]

# NOT 조건: 짝수가 아닌 요소 (홀수)
mask_not = ~(arr % 2 == 0)  # arr % 2 != 0 와 동일
print(arr[mask_not])  # [1 3 5 7 9]

# 복합 조건: 홀수이면서 5보다 큰 요소
complex_mask = (arr % 2 == 1) & (arr > 5)
print(arr[complex_mask])  # [7 9]
```

### 마스크로 값 변경하기

불리언 마스크를 사용하여 조건을 만족하는 요소의 값을 변경할 수 있다:

```python
# 마스크를 통한 값 변경
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 5보다 큰 요소를 모두 50으로 변경
arr[arr > 5] = 50
print(arr)  # [ 1  2  3  4  5 50 50 50 50 50]

# 조건에 따른 다른 값 할당
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr[arr % 2 == 0] = -1  # 짝수를 모두 -1로 변경
print(arr)  # [ 1 -1  3 -1  5 -1  7 -1  9 -1]

# np.where를 사용한 조건부 값 할당
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = np.where(arr % 2 == 0, -1, arr)  # 짝수는 -1로, 홀수는 원래값 유지
print(result)  # [ 1 -1  3 -1  5 -1  7 -1  9 -1]
```

### 마스킹을 활용한 데이터 분석 예제

마스킹은 실제 데이터 분석에서 매우 유용하다:

```python
# 데이터 분석 예제
# 학생들의 시험 점수 데이터
scores = np.array([
    # 수학, 과학, 영어, 국어, 사회 점수
    [85, 90, 75, 80, 95],  # 학생 1
    [65, 70, 85, 90, 75],  # 학생 2
    [95, 90, 85, 70, 60],  # 학생 3
    [75, 80, 65, 90, 85],  # 학생 4
    [80, 75, 90, 85, 80]   # 학생 5
])

# 1. 수학 점수가 80점 이상인 학생들의 모든 과목 점수
high_math = scores[scores[:, 0] >= 80]
print("수학 고득점자:", high_math)

# 2. 모든 과목이 80점 이상인 학생
all_high = scores[np.all(scores >= 80, axis=1)]
print("모든 과목 고득점자:", all_high)

# 3. 적어도 한 과목이 90점 이상인 학생
any_excellent = scores[np.any(scores >= 90, axis=1)]
print("한 과목 이상 우수한 학생:", any_excellent)

# 4. 특정 학생의 평균 이상인 과목 찾기
student_idx = 0  # 첫 번째 학생
student_scores = scores[student_idx]
student_avg = np.mean(student_scores)
above_avg = student_scores[student_scores > student_avg]
print(f"학생 {student_idx+1}의 평균 이상 점수:", above_avg)
```

### 마스킹과 인덱싱 결합하기

마스킹과 다른 인덱싱 방법을 결합해 복잡한 데이터 접근이 가능하다:

```python
# 마스킹과 다른 인덱싱 기법 결합
arr_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# 행 선택 후 마스킹
row_selection = arr_2d[1:3]       # 두 번째와 세 번째 행 선택
mask = row_selection > 7          # 7보다 큰 요소에 대한 마스크
print(row_selection[mask])        # [ 8  9 10 11 12]

# 마스킹으로 인덱스 배열 생성 후 인덱싱
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
idx = np.where(arr > 40)[0]       # 40보다 큰 요소의 인덱스
print(idx)                        # [4 5 6 7]
print(arr[idx])                   # [50 60 70 80]
```

마스킹은 NumPy에서 가장 강력한 데이터 선택 및 필터링 도구 중 하나로, 데이터 처리 과정에서 불필요한 반복문을 줄이고 코드를 간결하게 만드는 데 큰 도움이 된다.

## 2.4.4 NumPy 배열 연산

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

### NumPy의 axis 인자 이해하기

NumPy 함수의 `axis` 인자는 연산을 수행할 배열의 "축"을 지정한다. 이 개념은 처음에는 직관적이지 않을 수 있지만, 데이터 분석에서 중요하다:

```python
# axis 개념 이해를 위한 2차원 배열 예시
arr_2d = np.array([
    [1, 2, 3],  # 행 0
    [4, 5, 6],  # 행 1
    [7, 8, 9]   # 행 2
])  # 3x3 배열

# axis=0: 각 열을 따라 연산 (행 방향으로 축소)
col_sums = arr_2d.sum(axis=0)
print(col_sums)  # [12 15 18] - 각 열의 합

# axis=1: 각 행을 따라 연산 (열 방향으로 축소)
row_sums = arr_2d.sum(axis=1)
print(row_sums)  # [ 6 15 24] - 각 행의 합

# axis 인자 미지정 시 동작 (기본값: axis=None)
total_sum = arr_2d.sum()  # axis=None과 동일
print(total_sum)  # 45 (모든 요소의 합)

# 여러 예시로 axis=None의 동작 확인
arr_1d = np.array([1, 2, 3, 4])
print(arr_1d.sum())  # 10 (모든 요소의 합)

arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2x2 배열
print(arr_3d.sum())  # 36 (모든 요소의 합)

# 다른 집계 함수들의 동작 비교
print(f"최댓값(axis 미지정): {arr_2d.max()}")  # 9 (전체 최댓값)
print(f"최댓값(axis=0): {arr_2d.max(axis=0)}")  # [7 8 9] (각 열의 최댓값)
print(f"최댓값(axis=1): {arr_2d.max(axis=1)}")  # [3 6 9] (각 행의 최댓값)
```

#### axis 인자의 작동 원리

`axis` 인자를 지정하거나 지정하지 않을 때 NumPy 함수가 동작하는 방식을 다음과 같이 요약할 수 있다:

- **`axis=None` (기본값)**: 배열의 모든 요소를 평탄화(flatten)한 후 단일 연산을 수행하여 스칼라 값 반환
- **`axis=0`**: 첫 번째 차원을 따라 연산, 결과는 해당 차원이 제거된 형태
- **`axis=1`**: 두 번째 차원을 따라 연산, 결과는 해당 차원이 제거된 형태
- **`axis=(0,1)` 등 여러 축 지정**: 지정된 모든 축에 대해 연산 수행, 결과는 해당 차원들이 모두 제거된 형태

주의할 점은 `axis=None`과 모든 축을 명시적으로 나열한 경우(예: 3차원 배열에서 `axis=(0,1,2)`)는 최종 결과값이 동일할 수 있지만, 내부 계산 과정이 다를 수 있다는 점이다. `axis=None`은 먼저 배열을 1차원으로 평탄화한 후 연산하는 반면, 모든 축을 나열한 경우는 각 차원별로 순차적으로 연산할 수 있다.

### NumPy 전역 함수와 메서드 형식

NumPy는 많은 함수를 두 가지 방식으로 제공한다 - 전역 함수(`np.함수명()`)와 배열 메서드(`배열.함수명()`). 이 두 가지 방식의 차이점과 사용법을 이해하는 것이 중요하다:

```python
# 배열 생성
arr = np.array([4, 2, 7, 1])

# 1. 전역 함수 방식
max_val = np.max(arr)     # np.함수명(배열, 추가인자...)
argmax_idx = np.argmax(arr)
mean_val = np.mean(arr)

# 2. 메서드 방식
max_val = arr.max()       # 배열.함수명(추가인자...)
argmax_idx = arr.argmax()
mean_val = arr.mean()

# 둘 다 동일한 결과 산출
print(f"최댓값: {max_val}, 최댓값 위치: {argmax_idx}, 평균: {mean_val}")
```

#### 두 방식의 차이점

1. **유연성**:
   - 전역 함수: 다양한 입력 타입(리스트, 튜플 등)을 처리할 수 있으며, 추가 옵션이나 인자를 쉽게 추가 가능
   - 메서드: 이미 NumPy 배열인 경우에만 사용 가능하지만 코드가 더 간결해짐

2. **사용 패턴**:
   - 전역 함수: 여러 배열을 동시에 처리하거나 복잡한 연산에 유용

   ```python
   # 여러 배열에 대해 같은 연산 수행
   arr1 = np.array([1, 2, 3])
   arr2 = np.array([4, 5, 6])
   means = np.mean([arr1, arr2], axis=0)  # [2.5 3.5 4.5]
   ```

   - 메서드: 체이닝이나 객체 지향 코드에서 더 자연스러움

   ```python
   # 메서드 체이닝 예시
   result = ((arr - arr.mean()) / arr.std()).round(2)
   ```

3. **가용성**:
   - 일부 함수는 전역 함수로만 존재: `np.vstack()`, `np.concatenate()`, `np.linalg` 모듈의 함수들
   - 대부분의 기본 연산은 두 가지 형태로 모두 제공됨

#### 사용 가이드라인

- 기본적으로 둘 중 편리한 방식 사용 가능
- 파이프라인 스타일 코드에서는 메서드 형태가 유리
- 여러 배열을 동시에 처리하거나 NumPy 외부 객체를 다룰 때는 전역 함수가 더 적합
- 코드베이스 내 일관성을 유지하는 것이 중요

## 2.4.5 행렬 곱셈 연산자 `@`

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

## 2.4.6 `@` 연산자와 사용자 정의 클래스

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

## 2.4.7 브로드캐스팅(Broadcasting)

브로드캐스팅(Broadcasting)은 형태가 다른 배열 간의 연산을 가능하게 하는 NumPy의 강력한 기능이다. 이름 그대로 '작은 배열'을 '넓게(broad)' '변형/확장(casting)'하여 더 큰 배열과 연산할 수 있도록 해준다. 실제 메모리 사용 없이 작은 배열이 큰 배열의 형태에 맞게 확장된 것처럼 연산을 수행한다:

```python
# 스칼라와 배열의 연산
a = np.array([1, 2, 3, 4])
print(a + 10)         # [11 12 13 14] (모든 요소에 10 더함)
                      # 스칼라 10이 [10, 10, 10, 10]으로 확장된 것처럼 동작

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

### 브로드캐스팅 규칙

NumPy는 다음 규칙에 따라 브로드캐스팅을 수행한다:

1. **차원 맞추기**: 두 배열의 차원 수가 다를 경우, 부족한 차원에 1을 추가하여 차원 수를 맞춘다 (오른쪽 정렬)
2. **차원별 호환성 검사**: 각 차원별로 크기가 같거나 둘 중 하나가 1인 경우에만 호환된다
3. **크기 1 차원 확장**: 크기가 1인 차원은 다른 배열의 해당 차원 크기에 맞게 확장된다

```python
# 브로드캐스팅의 단계적 진행 과정
a = np.array([[1, 2, 3], [4, 5, 6]])   # 형태: (2, 3)
b = np.array([10, 20, 30])             # 형태: (3,)

# 1. 차원 맞추기: b는 (3,) → (1, 3)으로 확장 (차원 수를 맞춤)
# 2. 차원별 호환성 검사:
#    각 차원에서 다음 규칙 적용:
#    - 두 차원의 크기가 같으면 호환 가능
#    - 두 차원 중 하나가 1이면, 크기 1인 차원은 다른 차원의 크기로 확장 가능
#
#    첫 번째 차원: 배열 a = 2, 배열 b = 1 
#                → b의 첫 차원(1)이 a의 첫 차원(2)으로 확장 가능
#    두 번째 차원: 배열 a = 3, 배열 b = 3
#                → 두 배열의 차원 크기가 같으므로 호환 가능
#
# 3. 실제 확장: 배열 b는 개념적으로 다음과 같이 확장됨
#    [[10, 20, 30],   ← 첫 번째 행 (메모리 복사 없이 가상으로 확장)
#     [10, 20, 30]]   ← 두 번째 행 (메모리 복사 없이 가상으로 확장)
print(a + b)  # [[11 22 33], [14 25 36]]
```

### 브로드캐스팅 실패 케이스

브로드캐스팅이 불가능한 경우 `ValueError`가 발생한다. 주요 실패 케이스:

```python
# 케이스 1: 차원별 크기가 맞지 않고, 둘 다 1이 아닌 경우
a = np.ones((3, 4))   # 3x4 배열
b = np.ones((2, 4))   # 2x4 배열
try:
    result = a + b
except ValueError as e:
    print(f"오류: {e}")
    # 출력: "오류: operands could not be broadcast together with shapes (3,4) (2,4)"

# 케이스 2: 차원 수가 다르고, 차원별 크기가 호환되지 않는 경우
a = np.ones((3, 2))
b = np.ones((3, 4, 2))
try:
    result = a + b
except ValueError as e:
    print(f"오류: {e}")
    # 출력: "오류: operands could not be broadcast together with shapes (3,2) (3,4,2)"

# 실패와 성공 비교: 미묘한 차이
a = np.ones((3, 1, 5))  # 3x1x5 배열
b1 = np.ones((3, 4, 1)) # 3x4x1 배열 - 성공 케이스
b2 = np.ones((3, 4, 2)) # 3x4x2 배열 - 실패 케이스

# 성공: a(3,1,5)와 b1(3,4,1) → 결과(3,4,5)
# 각 차원 비교: 3=3, 1+4=5, 5+1=6 (1은 확장 가능)
print((a + b1).shape)  # (3, 4, 5)

try:
    # 실패: a(3,1,5)와 b2(3,4,2) → 마지막 차원이 5≠2이고 둘 다 1이 아님
    result = a + b2
except ValueError as e:
    print(f"오류: {e}")
    # 출력: "오류: operands could not be broadcast together with shapes (3,1,5) (3,4,2)"
```

브로드캐스팅은 코드를 간결하게 만들고 배열 연산의 효율성을 높이지만, 형태 호환성을 정확히 이해해야 의도치 않은 오류를 방지할 수 있다.

## 2.4.8 유용한 NumPy 함수

NumPy는 다양한 유용한 함수를 제공한다. 여기서는 NumPy가 제공하는 함수의 특징과 주요 함수들을 살펴본다.

### NumPy의 유니버설 함수 (Universal Functions, ufunc)

NumPy의 핵심 기능 중 하나는 '유니버설 함수(Universal Functions, ufunc)'이다. ufunc는 배열의 각 요소에 대해 동일한 연산을 병렬적으로 수행하도록 최적화된 함수이다. 이러한 함수들은 다음과 같은 특징을 가진다:

1. **C/Fortran으로 구현**: 내부적으로 저수준 언어로 작성되어 매우 빠른 속도로 동작
2. **SIMD 명령어 활용**: CPU의 단일 명령 다중 데이터(SIMD) 기능을 활용하여 병렬 처리
3. **메모리 캐시 최적화**: 데이터 접근 패턴이 메모리 캐시를 효율적으로 사용하도록 설계
4. **멀티코어 활용**: 일부 함수는 여러 CPU 코어를 활용해 연산 병렬화

ufunc는 전통적인 Python 루프보다 수십에서 수백 배 빠를 수 있으며, 특히 큰 배열에서 그 효과가 두드러진다:

```python
import numpy as np
import time

# 벡터화 성능 비교
size = 10000000
data = np.random.random(size)

# 1. Python 루프 방식 (느림)
start = time.time()
result1 = [x**2 for x in data]
python_time = time.time() - start

# 2. NumPy ufunc 방식 (빠름)
start = time.time()
result2 = np.square(data)  # 또는 data**2
numpy_time = time.time() - start

print(f"Python 루프 시간: {python_time:.4f}초")
print(f"NumPy ufunc 시간: {numpy_time:.4f}초")
print(f"속도 향상: {python_time/numpy_time:.1f}배")
```

위 코드에서 NumPy의 ufunc는 일반 Python 루프보다 훨씬 빠르게 실행된다. 이런 성능 향상은 데이터 과학과 수치 연산 분야에서 NumPy가 필수적인 도구가 된 주요 이유 중 하나이다.

아래 함수 요약표에 포함된 수많은 함수들 중 상당수(특히 수학 및 통계 함수 카테고리의 함수들)는 ufunc로 구현되어 있다. 이들 함수는 파이썬의 순수 구현보다 훨씬 빠른 성능을 제공하므로, 대용량 데이터 처리에 적극 활용하는 것이 좋다.

### 자주 사용되는 NumPy 함수 개요 및 메서드 요약

NumPy는 다양한 수학 함수와 응용 함수들을 제공한다. 간단한 예시를 통해 NumPy의 주요 함수들을 살펴보자:

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

# 랜덤 배열 생성 예시
print("\n# 다양한 랜덤 배열 생성:")
print("1. 균등분포 난수:")
print(np.random.rand(2, 3))    # 0~1 균등분포
# [[0.12345678 0.23456789 0.34567891]
#  [0.45678912 0.56789123 0.67891234]]

print("\n2. 표준정규분포 난수:")
# np.random.randn(2, 3)에서:
#   - 첫 번째 인자 2: 생성할 배열의 행(row) 개수
#   - 두 번째 인자 3: 생성할 배열의 열(column) 개수
#   - 결과: 2×3 크기의 행렬(matrix) 생성
print(np.random.randn(2, 3))   # 평균 0, 표준편차 1의 정규분포
# [[ 0.12345678 -0.23456789  1.34567891]  ← 첫 번째 행(3개 열)
#  [-0.45678912  0.56789123 -0.67891234]]  ← 두 번째 행(3개 열)

print("\n2-1. 사용자 지정 정규분포 난수:")
# np.random.normal(평균, 표준편차, 크기)를 사용하면 
# 원하는 평균과 표준편차를 가진 정규분포 난수 생성 가능
print("평균=5, 표준편차=2의 정규분포:")
print(np.random.normal(5, 2, (2, 3)))  # 평균 5, 표준편차 2의 정규분포
# [[ 5.42345678  1.76543211  7.34567891]
#  [ 6.98765432  4.56789123  3.32108766]]

# 참고: randn() 결과에 표준편차를 곱하고 평균을 더하는 방식으로도 구현 가능
print("평균=5, 표준편차=2의 정규분포 (randn 활용):")
print(5 + 2 * np.random.randn(2, 3))  # 동일한 분포 특성

print("\n3. 정수 난수:")
print(np.random.randint(0, 10, (2, 3)))  # 0~9 사이의 정수
# [[3 7 2]
#  [9 1 4]]

# 배열 조작 함수 계속...
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

다음은 NumPy에서 가장 자주 사용되는 기본 함수와 메서드들의 요약표이다:

#### 배열 생성 함수

| 함수 | 설명 | 예시 |
|------|------|------|
| `array()` | 배열 생성 | `np.array([1, 2, 3])` |
| `zeros()` | 모든 값이 0인 배열 생성 | `np.zeros((2, 3))` |
| `ones()` | 모든 값이 1인 배열 생성 | `np.ones((2, 3))` |
| `empty()` | 초기화되지 않은 배열 생성 | `np.empty((2, 3))` |
| `arange()` | 범위로 배열 생성 | `np.arange(10)` |
| `linspace()` | 균등하게 분할된 값으로 배열 생성 | `np.linspace(0, 1, 5)` |
| `random.rand()` | 균등분포 난수 배열 생성 | `np.random.rand(2, 3)` |
| `random.randn()` | 표준정규분포 난수 배열 생성 | `np.random.randn(2, 3)` |
| `random.randint()` | 지정 범위 내 정수 난수 배열 | `np.random.randint(0, 10, (3, 3))` |
| `eye()` | 단위행렬 생성 | `np.eye(3)` |
| `full()` | 지정된 값으로 채워진 배열 | `np.full((2, 2), 7)` |
| `meshgrid()` | 좌표 격자 배열 생성 | `np.meshgrid(np.arange(3), np.arange(2))` |

#### 배열 조작 함수

| 함수/메서드 | 설명 | 예시 |
|------------|------|------|
| `reshape()` | 배열 형태 변경 | `arr.reshape(3, 4)` |
| `flatten()` | 1차원 배열로 평탄화 | `arr.flatten()` |
| `transpose()` / `.T` | 배열 전치 | `arr.transpose()` 또는 `arr.T` |
| `concatenate()` | 배열 연결 | `np.concatenate([a, b])` |
| `vstack()` | 수직 방향으로 배열 쌓기 | `np.vstack([a, b])` |
| `hstack()` | 수평 방향으로 배열 쌓기 | `np.hstack([a, b])` |
| `stack()` | 새로운 축을 따라 배열 쌓기 | `np.stack([a, b], axis=0)` |
| `split()` | 배열을 여러 부분으로 분할 | `np.split(arr, 3)` |
| `hsplit()` | 수평 방향으로 배열 분할 | `np.hsplit(arr, 3)` |
| `vsplit()` | 수직 방향으로 배열 분할 | `np.vsplit(arr, 3)` |
| `tile()` | 배열 반복하여 새 배열 생성 | `np.tile(arr, (2, 3))` |
| `repeat()` | 요소를 반복하여 새 배열 생성 | `np.repeat(arr, 3)` |
| `sort()` | 배열 정렬 | `np.sort(arr)` 또는 `arr.sort()` |
| `newaxis` | 새 차원(축) 추가 | `arr[:, np.newaxis]` |

#### 수학 및 통계 함수

| 함수/메서드 | 설명 | 예시 |
|------------|------|------|
| `add()` / `+` | 요소별 덧셈 | `np.add(a, b)` 또는 `a + b` |
| `subtract()` / `-` | 요소별 뺄셈 | `np.subtract(a, b)` 또는 `a - b` |
| `multiply()` / `*` | 요소별 곱셈 | `np.multiply(a, b)` 또는 `a * b` |
| `divide()` / `/` | 요소별 나눗셈 | `np.divide(a, b)` 또는 `a / b` |
| `power()` / `**` | 요소별 거듭제곱 | `np.power(a, 2)` 또는 `a ** 2` |
| `sum()` | 합계 | `np.sum(arr)` 또는 `arr.sum()` |
| `mean()` | 평균 | `np.mean(arr)` 또는 `arr.mean()` |
| `std()` | 표준편차 | `np.std(arr)` 또는 `arr.std()` |
| `var()` | 분산 | `np.var(arr)` 또는 `arr.var()` |
| `min()` | 최솟값 | `np.min(arr)` 또는 `arr.min()` |
| `max()` | 최댓값 | `np.max(arr)` 또는 `arr.max()` |
| `argmin()` | 최솟값 인덱스 | `np.argmin(arr)` 또는 `arr.argmin()` |
| `argmax()` | 최댓값 인덱스 | `np.argmax(arr)` 또는 `arr.argmax()` |
| `median()` | 중앙값 | `np.median(arr)` |
| `percentile()` | 백분위수 | `np.percentile(arr, 75)` |
| `all()` | 모든 요소가 참인지 | `np.all(arr > 0)` 또는 `(arr > 0).all()` |
| `any()` | 하나라도 참인지 | `np.any(arr > 0)` 또는 `(arr > 0).any()` |

#### 선형대수 함수

| 함수 | 설명 | 예시 |
|------|------|------|
| `dot()` / `@` | 행렬 곱셈 | `np.dot(a, b)` 또는 `a @ b` |
| `matmul()` | 행렬 곱셈 | `np.matmul(a, b)` |
| `inner()` | 내적 | `np.inner(a, b)` |
| `outer()` | 외적 | `np.outer(a, b)` |
| `linalg.det()` | 행렬식 계산 | `np.linalg.det(arr)` |
| `linalg.inv()` | 역행렬 계산 | `np.linalg.inv(arr)` |
| `linalg.solve()` | 선형 방정식 풀이 | `np.linalg.solve(a, b)` |
| `linalg.eig()` | 고유값, 고유벡터 계산 | `np.linalg.eig(arr)` |
| `linalg.svd()` | 특이값 분해 | `np.linalg.svd(arr)` |
| `linalg.norm()` | 벡터 또는 행렬 노름 계산 | `np.linalg.norm(arr)` |

#### 배열 검색 및 변환 함수

| 함수 | 설명 | 예시 |
|------|------|------|
| `where()` | 조건에 따라 값 선택 | `np.where(arr > 0, arr, 0)` |
| `clip()` | 값을 지정 범위로 제한 | `np.clip(arr, 0, 1)` |
| `unique()` | 중복 요소 제거된 배열 반환 | `np.unique(arr)` |
| `expand_dims()` | 새 차원 추가 | `np.expand_dims(arr, axis=0)` |
| `squeeze()` | 1인 차원 제거 | `np.squeeze(arr)` |
| `astype()` | 배열 데이터 타입 변환 | `arr.astype(np.float64)` |
| `round()` | 반올림 | `np.round(arr, 2)` 또는 `arr.round(2)` |
| `flip()` | 지정 축을 따라 배열 뒤집기 | `np.flip(arr, axis=0)` |
| `isnan()` | NaN 값 체크 | `np.isnan(arr)` |
| `isfinite()` | 유한 값 체크 | `np.isfinite(arr)` |

이 함수들은 NumPy로 데이터 분석이나 과학 계산을 할 때 가장 자주 사용되는 핵심 함수와 메서드들이다. 상황에 맞게 적절한 함수를 선택하여 효율적인 코드를 작성할 수 있다.

## 고급 NumPy 기능

NumPy의 더 고급 기능들은 [2.5 NumPy 심화](./2_5_numpy_deep_dive.md) 문서에서 자세히 다룬다:

- [선형대수 연산](./2_5_numpy_deep_dive.md#251-선형대수-연산): 행렬 곱셈, 고유값, 선형 방정식 해법 등
- [텐서와 다차원 배열](./2_5_numpy_deep_dive.md#252-텐서와-numpy의-다차원-배열): 고차원 데이터 처리
- [전치행렬과 메모리 효율성](./2_5_numpy_deep_dive.md#253-전치행렬transpose과-메모리-효율성): 효율적인 배열 조작
- [데이터 타입과 타입 변환](./2_5_numpy_deep_dive.md#254-numpy-데이터-타입과-타입-변환): 메모리 사용 최적화와 성능 향상

---
> [목차로 돌아가기](../../README.md) | [이전: 딥러닝 라이브러리 개요](./2_3_deep_learning_libraries.md) | [다음: NumPy 심화](./2_5_numpy_deep_dive.md)
