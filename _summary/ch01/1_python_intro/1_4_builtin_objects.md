# 1.4 내장 전역 객체

> [목차로 돌아가기](../../README.md) | [이전: 객체 시스템과 객체 지향 프로그래밍 지원](./1_3_oop_system.md) | [다음: 타입 힌트 심화: 값과 타입의 경계](./1_5_type_hint_deep_dive.md)

## 1.4.1 None 객체

### a. None의 정의와 특성

`None`은 파이썬에서 '값이 없음'을 나타내는 특별한 객체이다. `None`은 파이썬의 싱글톤(singleton) 객체로, 시스템 전체에 단 하나만 존재한다. 값의 부재, 초기화되지 않은 변수, 또는 함수에서 명시적인 반환값이 없을 때 사용된다.
  
`None`의 주요 특징:
  
1. `NoneType`이라는 고유한 타입을 가진다:
  
   ```python
   print(type(None))  # <class 'NoneType'>
   ```
  
2. 메모리에 하나만 존재하는 싱글톤 객체이다:
  
   ```python
   a = None
   b = None
   print(a is b)  # True - 항상 같은 객체를 참조
   ```
  
3. 불리언 컨텍스트에서 `False`로 평가된다:
  
   ```python
   print(bool(None))  # False
   
   if None:
       print("실행되지 않음")
   else:
       print("None은 False로 평가됨")  # 이 부분이 출력됨
   ```
  
4. 기본 반환값으로 사용된다:
  
   ```python
   def func_without_return():
       pass
       
   result = func_without_return()
   print(result)  # None
   print(result is None)  # True
   ```
  
### b. None의 활용

`None`은 '빈 값'을 나타내는 다른 객체들(빈 문자열 `""`, 빈 리스트 `[]`, 숫자 `0`)과는 다르다. `None`은 값 자체가 없음을 의미한다:
  
```python
empty_str = ""
empty_list = []
zero = 0
none_value = None
  
print(empty_str == None)  # False
print(empty_list == None)  # False
print(zero == None)  # False
print(none_value is None)  # True
```
  
`None` 값을 비교할 때는 항상 `is` 연산자를 사용해야 한다:
  
```python
# 권장 방식
if value is None:
    print("값이 None입니다")

if value is not None:
    print("값이 None이 아닙니다")
    
# 권장하지 않는 방식
if value == None:  # 동작은 하지만 is None을 사용하는 것이 더 명시적이고 효율적임
    print("값이 None입니다")
```

### c. None을 활용한 패턴

함수에서 `None`을 기본값으로 사용하는 일반적인 패턴:

```python
def process_data(data=None):
    if data is None:
        data = []  # 기본값으로 빈 리스트 생성
    # 데이터 처리...
    return data

# 인자 없이 호출
result = process_data()  # 새 빈 리스트 반환
```

선택적 반환값으로 `None` 사용:

```python
def find_user(user_id):
    # 사용자가 없으면 None 반환
    if user_id <= 0:  # 예시 조건
        return None
    # 사용자가 있으면 사용자 객체 반환
    return {"id": user_id, "name": f"User {user_id}"}  # 예시 사용자 객체

# 반환값 확인
user = find_user(123)
if user is not None:
    print(f"사용자를 찾았습니다: {user['name']}")
else:
    print("사용자를 찾을 수 없습니다")
```

## 1.4.2 불리언 객체 - True와 False

### a. True와 False의 특성

`True`와 `False`는 파이썬의 불리언(boolean) 타입을 나타내는 내장 상수이다. 이들은 각각 논리적 참과 거짓을 표현한다:

```python
# bool 타입 확인
print(type(True))  # <class 'bool'>
print(type(False))  # <class 'bool'>

# bool은 int의 서브클래스
print(issubclass(bool, int))  # True

# 정수 값으로 사용 가능
print(True + 1)  # 2
print(False * 10)  # 0
```

`bool` 타입은 `int`의 서브클래스로, `True`는 1, `False`는 0과 같다:

```python
print(True == 1)  # True
print(False == 0)  # True

# 그러나 식별자는 다름
print(True is 1)  # False
print(False is 0)  # False
```

### b. 불리언 평가 규칙

파이썬에서는 모든 객체가 불리언 컨텍스트에서 `True` 또는 `False`로 평가된다:

```python
# False로 평가되는 값들
print(bool(None))  # False
print(bool(0))     # False
print(bool(0.0))   # False
print(bool(""))    # False
print(bool([]))    # False
print(bool({}))    # False
print(bool(set())) # False

# True로 평가되는 값들
print(bool(1))       # True
print(bool(-1))      # True
print(bool(0.1))     # True
print(bool("text"))  # True
print(bool([0]))     # True
print(bool({"key": None}))  # True
```

### c. 불리언 연산

기본 불리언 연산자:

```python
# and, or, not 연산자
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# 단축 평가(short-circuit evaluation)
print(False and print("실행되지 않음"))  # False
print(True or print("실행되지 않음"))    # True

# 첫 번째 참/거짓 값 반환
print(0 or None or [] or "first" or 42)  # "first" (첫 번째 참 값)
print(1 and "text" and 0 and True)      # 0 (첫 번째 거짓 값)
```

## 1.4.3 Ellipsis 객체 (...)

### a. Ellipsis의 기본 용도

`...`(Ellipsis)는 파이썬의 싱글톤 객체로 여러 컨텍스트에서 사용된다:

```python
# Ellipsis 객체 확인
print(...)  # Ellipsis
print(type(...))  # <class 'ellipsis'>
print(... is Ellipsis)  # True
```

### b. 코드 스텁과 플레이스홀더로 활용

`pass`와 유사하게 미구현 코드 블록을 표시할 수 있다:

```python
def function_to_implement_later():
    ...  # 추후 구현 예정

class ClassToImplementLater:
    ...  # 추후 구현 예정
```

### c. 타입 힌트에서의 활용

타입 힌트에서 가변 길이 튜플이나 미확정 반환 타입을 나타낼 수 있다:

```python
from typing import Tuple

# 임의 개수의 정수를 가진 튜플
IntTuple = Tuple[int, ...]

def process_ints(*args: int) -> IntTuple:
    return args
    
# 아직 결정되지 않은 반환 타입
def func_with_unknown_return_type(x: int) -> ...:
    return x * 2
```

### d. 다차원 배열 인덱싱에서의 활용

과학 계산 라이브러리(NumPy 등)에서는 다차원 배열 인덱싱에 활용된다:

```python
# NumPy 예시 (실제 실행하려면 NumPy 설치 필요)
import numpy as np

# 3차원 배열 생성
array = np.zeros((3, 4, 5))  # 3x4x5 크기의 0으로 채워진 배열

# 모든 첫 번째 차원과 두 번째 차원에 대해 세 번째 차원의 첫 요소 선택
first_elements = array[..., 0]
# 위 코드는 array[:, :, 0]과 동일
```

## 1.4.4 __debug__ 상수

`__debug__`는 파이썬 프로그램이 최적화 모드로 실행되는지 여부를 나타내는 내장 상수이다:

```python
# 일반 실행 모드에서는 True
print(__debug__)  # True (보통의 경우)

# -O 또는 -OO 플래그로 실행하면 False가 됨
# python -O script.py 실행 시: False

# 조건부 디버깅 코드
if __debug__:
    print("디버그 모드에서만 실행되는 코드")
    # 개발 중 검증 코드...
```

`__debug__`가 `True`일 때만 실행되는 `assert` 문:

```python
def divide(a, b):
    # b가 0이 아니어야 함 (디버그 모드에서만 검사)
    assert b != 0, "0으로 나눌 수 없습니다"
    return a / b

try:
    divide(10, 0)
except AssertionError as e:
    print(f"검증 오류: {e}")  # 검증 오류: 0으로 나눌 수 없습니다

# 최적화 모드(-O)에서는 assert가 완전히 무시됨
# 따라서 중요한 검증은 assert 대신 if 문을 사용해야 함
```

## 1.4.5 NotImplemented 객체

### a. NotImplemented의 용도

`NotImplemented`는 이항 특수 메서드(binary special method)가 주어진 피연산자 조합에 대한 연산을 지원하지 않음을 나타내는 특수 상수이다:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    # 다른 Vector 객체와만 더할 수 있음
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented  # 다른 타입과의 연산은 지원하지 않음
        
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v = Vector(1, 2)
print(v + Vector(3, 4))  # Vector(4, 6)

# NotImplemented 반환 시 파이썬은 다른쪽 피연산자의 __radd__ 호출을 시도
try:
    print(v + 1)  # TypeError: unsupported operand type(s) for +: 'Vector' and 'int'
except TypeError as e:
    print(f"오류: {e}")
```

### b. NotImplemented와 NotImplementedError의 차이

`NotImplemented`와 `NotImplementedError`는 다른 목적을 가진다:

```python
# NotImplemented는 특수 메서드에서 반환되는 상수
print(type(NotImplemented))  # <class 'NotImplementedType'>

# NotImplementedError는 추상 메서드가 구현되지 않았을 때 발생하는 예외
class AbstractBase:
    def abstract_method(self):
        raise NotImplementedError("이 메서드는 자식 클래스에서 구현해야 합니다")

class Concrete(AbstractBase):
    # abstract_method를 구현하지 않음
    pass

try:
    obj = Concrete()
    obj.abstract_method()  # 구현되지 않은 메서드 호출
except NotImplementedError as e:
    print(f"오류: {e}")  # 오류: 이 메서드는 자식 클래스에서 구현해야 합니다
```

> [목차로 돌아가기](../../README.md) | [이전: 객체 시스템과 객체 지향 프로그래밍 지원](./1_3_oop_system.md) | [다음: 타입 힌트 심화: 값과 타입의 경계](./1_5_type_hint_deep_dive.md)
