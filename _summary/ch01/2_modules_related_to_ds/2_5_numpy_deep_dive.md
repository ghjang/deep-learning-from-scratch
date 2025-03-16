# 2.5 NumPy 심화

> [목차로 돌아가기](../../README.md) | [이전: NumPy 인트로](./2_4_numpy_intro.md) | [다음: matplotlib 인트로](./2_6_matplotlib_intro.md)

이 문서에서는 NumPy의 고급 기능과 심화 주제에 대해 다룹니다. [NumPy 기본 개요](./2_4_numpy_intro.md)를 이미 숙지한 상태에서 살펴보는 것이 좋습니다.

## 2.5.1 선형대수 연산

NumPy는 기본적인 선형대수 연산을 지원한다:

```python
import numpy as np

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

## 2.5.2 텐서와 NumPy의 다차원 배열

'**텐서(tensor)**'는 벡터와 행렬을 일반화한 다차원 데이터 구조로, 임의의 차원 수를 가질 수 있다. 텐서를 다룰 때 [reshape 메서드](./2_4_numpy_intro.md#reshape-메서드와--1-인수-활용)와 [전치 연산](#253-전치행렬transpose과-메모리-효율성)이 자주 사용된다:

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

## 2.5.3 전치행렬(Transpose)과 메모리 효율성

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

2차원 행렬에서 전치는 단순히 행과 열을 교환하는 것이지만, 3차원 이상의 텐서에서 전치는 '차원 순서의 재배열'을 의미한다. 이러한 고차원 데이터 조작에 앞서 [텐서의 개념](#252-텐서와-numpy의-다차원-배열)을 이해하는 것이 중요하다:

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

## 2.5.4 NumPy 데이터 타입과 타입 변환

NumPy는 다양한 수치 데이터 타입을 제공하며, 이는 메모리 사용량과 연산 속도에 큰 영향을 미친다. 적절한 데이터 타입 선택은 효율적인 계산을 위해 중요하다:

### a. 데이터 타입 추론 규칙

NumPy 배열을 생성할 때 dtype 매개변수를 지정하지 않으면, NumPy는 다음과 같은 규칙에 따라 자동으로 데이터 타입을 추론한다:

```python
# 데이터 타입 추론 예시
import numpy as np

# 1. 정수만 포함된 경우: 시스템에 따라 int32 또는 int64 (대부분 64비트 시스템에서는 int64)
int_array = np.array([1, 2, 3])
print(f"정수 배열: {int_array.dtype}")  # int64 (64비트 시스템 기준)

# 2. 실수가 하나라도 포함된 경우: float64
mixed_array = np.array([1, 2, 3.0])
print(f"실수 포함 배열: {mixed_array.dtype}")  # float64

# 3. 불리언 값만 포함된 경우: bool
bool_array = np.array([True, False, True])
print(f"불리언 배열: {bool_array.dtype}")  # bool

# 4. 복소수가 포함된 경우: complex128
complex_array = np.array([1, 2+3j])
print(f"복소수 포함 배열: {complex_array.dtype}")  # complex128

# 5. 다양한 정수 타입 혼합 시 상향 조정
mixed_ints = np.array([np.int8(1), np.int16(2), np.int32(3)])
print(f"혼합 정수 배열: {mixed_ints.dtype}")  # 최소한 int32 이상으로 상향

# 6. 특수 함수들의 기본값
print(f"np.zeros 기본 타입: {np.zeros(3).dtype}")  # float64
print(f"np.ones 기본 타입: {np.ones(3).dtype}")    # float64
print(f"np.empty 기본 타입: {np.empty(3).dtype}")  # float64

# 7. 타입 범위 초과 값은 손실 없이 더 큰 타입으로 상향 조정
large_values = np.array([2**31])  # int32 범위 초과
print(f"큰 정수 배열: {large_values.dtype}")  # int64

# 8. 문자열은 가장 긴 문자열을 수용할 수 있는 타입으로 선택
str_array = np.array(['a', 'ab', 'abc'])
print(f"문자열 배열: {str_array.dtype}")  # <U3 (유니코드 문자열, 최대 3글자)
```

**주요 규칙 요약:**

1. **입력 데이터에 기반한 추론**:
   - 정수만 포함 → int64 (64비트 시스템) 또는 int32 (32비트 시스템)
   - 실수 포함 → float64  
   - 복소수 포함 → complex128
   - 불리언 값만 포함 → bool
   - 문자열 → 유니코드 문자열(U) 또는 바이트 문자열(S)

2. **최소 공통 타입**: 다양한 타입이 섞여 있으면 모든 값을 정확히 표현할 수 있는 가장 작은 타입으로 상향 조정

3. **특수 배열 생성 함수**: `np.zeros()`, `np.ones()`, `np.empty()` 등의 함수는 dtype 인수가 없으면 기본적으로 float64 타입을 사용

4. **범위 초과 보호**: 제공된 데이터가 기본 타입의 범위를 초과하면, 자동으로 더 큰 타입으로 상향 조정

이러한 규칙을 이해하면 예상치 못한 타입 관련 문제를 방지하고, 필요한 경우 명시적으로 dtype을 지정할 수 있다.

### b. 주요 NumPy 데이터 타입

```python
# NumPy 기본 데이터 타입
import numpy as np

# 정수형 데이터 타입
int_types = [
    np.int8,    # -128 ~ 127 (1바이트)
    np.int16,   # -32,768 ~ 32,767 (2바이트)
    np.int32,   # -2^31 ~ 2^31-1 (4바이트)
    np.int64    # -2^63 ~ 2^63-1 (8바이트)
]

# 부호 없는 정수형
uint_types = [
    np.uint8,   # 0 ~ 255 (1바이트) - 이미지 처리에 자주 사용
    np.uint16,  # 0 ~ 65,535 (2바이트)
    np.uint32,  # 0 ~ 2^32-1 (4바이트)
    np.uint64   # 0 ~ 2^64-1 (8바이트)
]

# 부동소수점 타입
float_types = [
    np.float16,  # 반정밀도 (2바이트)
    np.float32,  # 단정밀도 (4바이트)
    np.float64,  # 배정밀도 (8바이트) - 기본 float 타입
]

# 기타 데이터 타입
other_types = [
    np.bool_,    # True/False 값 저장
    np.complex64, # 복소수 (실수부와 허수부 각각 32비트)
    np.complex128 # 복소수 (실수부와 허수부 각각 64비트)
]

# 배열 생성 시 데이터 타입 지정
int_array = np.array([1, 2, 3], dtype=np.int8)
float_array = np.array([1.0, 2.5, 3.7], dtype=np.float32)
uint_array = np.zeros((5, 5), dtype=np.uint8)  # 이미지 형식으로 자주 사용
```

### c. 데이터 타입 확인 및 변환

```python
# 데이터 타입 확인
arr = np.array([1, 2, 3])
print(arr.dtype)  # int64 (64비트 시스템 기준 기본값)

# astype 메서드로 타입 변환
float_arr = arr.astype(np.float32)
print(float_arr.dtype)  # float32
print(float_arr)  # [1. 2. 3.]

# 다양한 변환 방법
uint8_arr = np.array([255, 128, 0], dtype=np.uint8)
int16_arr = uint8_arr.astype(np.int16)  # uint8 -> int16
float_arr = uint8_arr.astype(float)     # uint8 -> float64

# 타입 생성자 사용
uint8_from_float = np.uint8([1.5, 2.5, 3.9])  # 소수점 버림
print(uint8_from_float)  # [1 2 3]
```

### d. 데이터 타입 변환의 응용 사례

#### 이미지 처리

NumPy 배열과 이미지 처리 라이브러리(PIL/Pillow, OpenCV 등) 간의 상호 변환에는 데이터 타입이 중요하다:

```python
import numpy as np
from PIL import Image

# 그레이스케일 이미지 생성 (0-255 값의 2D 배열)
img_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

# NumPy 배열 -> PIL 이미지
img = Image.fromarray(img_array)
img.save('grayscale.png')

# PIL 이미지 -> NumPy 배열
loaded_img = Image.open('grayscale.png')
loaded_array = np.array(loaded_img)  # 자동으로 uint8 타입으로 변환

# RGB 이미지 (3채널)
rgb_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
rgb_img = Image.fromarray(rgb_array)
rgb_img.save('rgb_image.png')
```

#### 메모리 최적화

적절한 데이터 타입 선택으로 메모리 사용량을 크게 줄일 수 있다:

```python
# 큰 배열의 메모리 사용량 비교
arr_float64 = np.ones(10000000, dtype=np.float64)  # 8바이트 × 1천만 = ~80MB
arr_float32 = np.ones(10000000, dtype=np.float32)  # 4바이트 × 1천만 = ~40MB
arr_uint8 = np.ones(10000000, dtype=np.uint8)      # 1바이트 × 1천만 = ~10MB

print(f"float64 메모리: {arr_float64.nbytes / 1048576:.2f} MB")
print(f"float32 메모리: {arr_float32.nbytes / 1048576:.2f} MB")
print(f"uint8 메모리: {arr_uint8.nbytes / 1048576:.2f} MB")

# 데이터 값 범위가 제한적인 경우, 작은 데이터 타입 사용이 유리
```

### e. 데이터 타입 변환 시 주의사항

```python
# 오버플로우 문제
small_int = np.array([200, 300], dtype=np.uint8)  # 최대 255까지 저장 가능
print(small_int)  # [200  44] - 300은 8비트 범위를 초과하여 오버플로 발생 (300-256=44)

# 타입 변환 시 정밀도 손실
x = np.array([1234.56789], dtype=np.float64)
x_float32 = x.astype(np.float32)
x_float16 = x.astype(np.float16)
print(f"원본(float64): {x[0]}")           # 1234.56789
print(f"float32로 변환: {x_float32[0]}")   # 약간의 정밀도 손실 가능
print(f"float16으로 변환: {x_float16[0]}")  # 상당한 정밀도 손실

# 부호 있는/없는 정수 간 변환
signed = np.array([-5, 10, 100], dtype=np.int8)
unsigned = signed.astype(np.uint8)
print(unsigned)  # [251  10 100] - 음수 값이 큰 양수로 해석됨 (-5 → 251)
```

### f. 디스크 공간 최적화를 위한 타입 변환

NumPy는 디스크 공간 절약을 위한 특수 파일 형식도 제공한다:

```python
# 큰 데이터 배열 생성
large_array = np.random.rand(1000, 1000)  # float64 배열 (~8MB)

# 표준 방식으로 저장
np.save('float64_array.npy', large_array)  # 원본 형식 그대로 저장

# 압축하여 저장
np.savez_compressed('float64_array_compressed.npz', array=large_array)

# 타입 변환 후 저장
float16_array = large_array.astype(np.float16)  # 정밀도를 희생하고 크기 절반으로 줄임
np.save('float16_array.npy', float16_array)

import os
print(f"float64 원본: {os.path.getsize('float64_array.npy') / 1048576:.2f} MB")
print(f"압축 버전: {os.path.getsize('float64_array_compressed.npz') / 1048576:.2f} MB")
print(f"float16 버전: {os.path.getsize('float16_array.npy') / 1048576:.2f} MB")
```

적절한 데이터 타입을 선택하고 필요에 따라 타입 변환을 수행하면 메모리 사용량 최적화, 연산 속도 향상, 타 라이브러리와의 호환성 개선 등 다양한 이점을 얻을 수 있다.

## 2.5.5 함수형 프로그래밍과 유사한 NumPy 연산

파이썬의 함수형 프로그래밍에서 자주 사용되는 `map`, `filter`, `reduce` 함수와 유사한 작업을 NumPy에서는 더 효율적인 방식으로 수행할 수 있다. 이러한 벡터화된 연산은 루프를 피하고 성능을 향상시킨다.

### a. Map 유사 기능: 배열 요소별 변환

파이썬의 `map()`처럼 배열의 모든 요소에 함수를 적용하는 방법:

```python
import numpy as np
from math import sin

# 기본적인 방법: NumPy의 유니버설 함수(ufunc) 사용
arr = np.array([0, np.pi/4, np.pi/2, np.pi])
sin_values = np.sin(arr)  # [0.0, 0.7071, 1.0, 0.0]

# 일반 함수를 벡터화하기
def custom_function(x):
    return x**2 if x < 0.5 else x**3

vectorized_func = np.vectorize(custom_function)
result = vectorized_func(np.array([0.1, 0.6, 0.4, 0.8]))
print(result)  # [0.01 0.216 0.16 0.512]

# 배열의 특정 축을 따라 함수 적용
def row_sum_and_mean(row):
    return np.sum(row), np.mean(row)

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_results = np.apply_along_axis(row_sum_and_mean, 1, matrix)
print(row_results)
# [[6.  2.]
#  [15. 5.]
#  [24. 8.]]
```

### b. Filter 유사 기능: 함수형 관점에서의 마스킹

파이썬의 `filter()`처럼 조건을 만족하는 요소만 선택하는 것은 NumPy에서 불리언 마스킹으로 구현된다. 기본적인 마스킹은 [NumPy 인트로의 마스킹 섹션](./2_4_numpy_intro.md#243-배열-마스킹boolean-masking)에서 다루었으며, 여기서는 함수형 프로그래밍 관점에서의 고급 사용법을 알아보자:

```python
# 함수형 스타일로 필터 연산 캡슐화하기
def filter_array(arr, condition_func):
    """함수형 스타일의 배열 필터링"""
    mask = condition_func(arr)
    return arr[mask]

# 필터 함수 예시
is_even = lambda x: x % 2 == 0
is_positive = lambda x: x > 0
is_in_range = lambda x, min_val, max_val: (x >= min_val) & (x <= max_val)

# 사용 예시
arr = np.array([-5, -2, 0, 3, 7, 8, 10])
print(filter_array(arr, is_even))  # [-2  0  8 10]
print(filter_array(arr, is_positive))  # [3 7 8 10]
print(filter_array(arr, lambda x: is_in_range(x, -3, 5)))  # [-2  0  3]

# 복잡한 조건 조합
complex_filter = lambda x: (x < -3) | ((x >= 0) & (x <= 5))
print(filter_array(arr, complex_filter))  # [-5  0  3]
```

고급 필터링 기법:

```python
# 구조적 데이터에 대한 필터링
data = np.array([
    (1, 'A', 3.5),
    (2, 'B', 2.7),
    (3, 'A', 1.5),
    (4, 'C', 4.0)
], dtype=[('id', int), ('category', 'U1'), ('value', float)])

# 특정 필드 기준으로 필터링
category_filter = data['category'] == 'A'
filtered_data = data[category_filter]
print(filtered_data)  # [(1, 'A', 3.5), (3, 'A', 1.5)]

# 다중 조건
complex_filter = (data['category'] == 'A') & (data['value'] > 2.0)
print(data[complex_filter])  # [(1, 'A', 3.5)]
```

### c. Reduce 유사 기능: 배열을 단일 값으로 축소

파이썬의 `functools.reduce()`처럼 배열의 요소를 누적 집계하는 방법:

```python
# 기본 축소 함수들
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))     # 15 (모든 요소의 합)
print(np.prod(arr))    # 120 (모든 요소의 곱)
print(np.mean(arr))    # 3.0 (평균)
print(np.max(arr))     # 5 (최댓값)
print(np.min(arr))     # 1 (최솟값)

# 다차원 배열에서 축 지정 축소
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_sums = np.sum(matrix, axis=1)  # 행별 합계
print(row_sums)  # [ 6 15 24]

col_means = np.mean(matrix, axis=0)  # 열별 평균
print(col_means)  # [4. 5. 6.]

# 누적 축소 함수
print(np.cumsum(arr))  # [ 1  3  6 10 15] (누적 합)
print(np.cumprod(arr))  # [  1   2   6  24 120] (누적 곱)
```

사용자 정의 축소 연산 - `np.apply_over_axes()` 및 `np.reduce()`:

```python
# 사용자 정의 함수를 축소 연산으로 사용
def custom_reduce(x, y):
    return x * y if x > y else x + y

# 보통은 내장 함수로 충분하지만, 복잡한 경우 루프 사용 필요
result = arr[0]
for i in range(1, len(arr)):
    result = custom_reduce(result, arr[i])
print(result)

# 유니버설 함수에 대한 np.reduce 사용
sum_reduce = np.add.reduce(arr)  # np.sum(arr)과 동일
print(sum_reduce)  # 15

# np.apply_over_axes - 여러 축에 대해 함수 적용
def max_and_sum(arr, axis):
    return np.array([np.max(arr, axis=axis), np.sum(arr, axis=axis)])

cube = np.arange(24).reshape(2, 3, 4)
result = np.apply_over_axes(max_and_sum, cube, [1, 2])
print(result.shape)  # 계산 결과의 형태
```

### d. 연산 결합 사례

NumPy는 이러한 map, filter, reduce 유사 기능을 결합하여 복잡한 데이터 처리 파이프라인을 구축할 수 있다:

```python
# 다단계 데이터 처리 예시
data = np.random.normal(0, 1, size=1000)  # 1000개의 정규분포 난수 생성

# 1. 필터링: 이상치 제거 (표준편차의 2배 범위 내 값만 사용)
mask = np.abs(data) < 2.0
filtered_data = data[mask]

# 2. 변환: 값 정규화 (0-1 사이로 스케일링)
min_val, max_val = filtered_data.min(), filtered_data.max()
normalized = (filtered_data - min_val) / (max_val - min_val)

# 3. 축소: 통계량 계산
stats = {
    'mean': np.mean(normalized),
    'median': np.median(normalized),
    'std': np.std(normalized),
    'q1': np.percentile(normalized, 25),
    'q3': np.percentile(normalized, 75)
}

print(f"처리된 데이터 개수: {len(filtered_data)}")
print(f"통계 정보: {stats}")
```

이러한 NumPy의 벡터화된 함수형 연산 기능은 데이터 분석 및 과학 계산에서 반복문 대신 사용할 수 있어 효율적인 코드 작성이 가능하다. 특히 대용량 데이터를 다룰 때 속도와 메모리 효율성 측면에서 큰 이점을 제공한다.

---
> [목차로 돌아가기](../../README.md) | [이전: NumPy 인트로](./2_4_numpy_intro.md) | [다음: matplotlib 인트로](./2_6_matplotlib_intro.md)
