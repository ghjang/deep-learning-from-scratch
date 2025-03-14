# 1.8 기본 데이터 타입

> [목차로 돌아가기](../../README.md) | [이전: 함수형 프로그래밍 요소](./1_7_functional_programming.md)

## 1.8.1 가변성과 불변성

파이썬의 데이터 타입은 가변(mutable)과 불변(immutable) 타입으로 나뉜다. 이 구분은 객체 생성 후 내용을 변경할 수 있는지에 따른 분류이다.

가변성과 불변성에 대한 자세한 내용은 [1.7.5 파이썬의 가변성과 불변성](./1_7_functional_programming.md#175-파이썬의-가변성mutability과-불변성immutability)을 참조하라.

## 1.8.2 파이썬 데이터 타입 개요

파이썬은 다양한 내장 데이터 타입을 제공하며, 이들은 가변성과 사용 목적에 따라 분류할 수 있다.

### a. 불변(Immutable) 타입

객체 생성 후 내용을 변경할 수 없는 타입:

- **숫자 타입**
  - `int`: 정수 (예: `42`, `-7`, `0`)
  - `float`: 부동소수점 수 (예: `3.14`, `-0.001`, `2.5e3`)
  - `complex`: 복소수 (예: `3+4j`)

- **문자열 및 바이트**
  - `str`: 문자열 (예: `"hello"`, `'world'`)
  - `bytes`: 바이트 시퀀스 (예: `b"hello"`)

- **컬렉션**
  - `tuple`: 불변 시퀀스 (예: `(1, 2, 3)`)
  - `frozenset`: 불변 집합 (예: `frozenset([1, 2, 3])`)

- **기타**
  - `bool`: 불리언 값 (`True` 또는 `False`)
  - `NoneType`: `None` 값을 위한 타입

### b. 가변(Mutable) 타입

객체 생성 후 내용을 변경할 수 있는 타입:

- **컬렉션**
  - `list`: 가변 시퀀스 (예: `[1, 2, 3]`)
  - `dict`: 키-값 쌍 매핑 (예: `{"name": "Kim", "age": 30}`)
  - `set`: 해시 가능한 고유 요소의 집합 (예: `{1, 2, 3}`)

- **기타**
  - `bytearray`: 가변 바이트 시퀀스 (예: `bytearray([65, 66, 67])`)

### c. 기능별 분류

사용 목적과 기능에 따른 분류:

- **기본 타입**: `int`, `float`, `complex`, `bool`, `None`
- **시퀀스 타입**: `str`, `bytes`, `bytearray`, `list`, `tuple`  
- **매핑 타입**: `dict`
- **집합 타입**: `set`, `frozenset`
- **바이너리 타입**: `bytes`, `bytearray`

이어지는 섹션에서는 이러한 데이터 타입들에 대해 자세히 살펴본다.

## 1.8.3 문자열 (`str`)

### a. 문자열의 기본 개념

문자열은 큰따옴표나 작은따옴표로 감싸서 표현할 수 있다:

```python
# 문자열 정의
single_quoted = 'Hello, World!'
double_quoted = "Hello, World!"
print(single_quoted)  # Hello, World!
print(double_quoted)  # Hello, World!
```

### b. 여러 줄 문자열

삼중 따옴표(`"""` 또는 `'''`)를 사용하여 여러 줄에 걸친 문자열을 표현할 수 있다:

```python
multiline = """
    Hello,
    World!
"""
print(multiline)

# 앞쪽 공백 제거
trimmed = """\
    Hello,
    World!
"""
print(trimmed)  # '    Hello,\n    World!'
```

**여러 줄 문자열과 들여쓰기**: 역슬래시(`\`)를 사용하면 들여쓰기 처리를 제어할 수 있다:

- 기본적으로 여러 줄 문자열의 첫 줄 이후에 나오는 모든 공백은 문자열에 포함된다
- `"""\`와 같이 열린 따옴표 뒤에 바로 역슬래시를 붙이면 첫 줄의 줄바꿈이 무시된다
- 이 방식으로 들여쓰기를 코드에 맞게 유지하면서도 출력에 불필요한 공백이 포함되는 것을 방지할 수 있다

**docstring**은 함수, 클래스, 모듈 등의 설명을 담는 특별한 형태의 문자열로, 주로 삼중 따옴표로 작성한다:

```python
def my_function():
    """이 함수는 어떤 작업을 수행합니다."""
    pass
```

### c. 문자열 메서드와 연산

문자열은 불변 객체이지만 다양한 메서드를 통해 변환된 새 문자열을 생성할 수 있다:

```python
text = "Hello, Python"

# 대소문자 변환
print(text.upper())       # HELLO, PYTHON
print(text.lower())       # hello, python
print(text.title())       # Hello, Python

# 검색 및 치환
print(text.find("Python"))  # 7 (시작 인덱스)
print(text.replace("Python", "World"))  # Hello, World

# 문자열 분할 및 결합
words = text.split(", ")
print(words)              # ['Hello', 'Python']
print(", ".join(words))   # Hello, Python

# 공백 제거
text_with_spaces = "  Hello  "
print(text_with_spaces.strip())   # "Hello"
```

## 1.8.4 숫자 타입 (`int`, `float`)

### a. 정수형(`int`)

파이썬 3에서 정수는 크기 제한이 없으며, 메모리가 허용하는 한 무제한으로 큰 정수를 표현할 수 있다:

```python
# 정수 리터럴
a = 42
b = -7
c = 0

# 매우 큰 정수도 처리 가능
big_num = 123456789123456789123456789
print(type(big_num))  # <class 'int'>
print(big_num * 2)    # 246913578246913578246913578

# 진법 표현
binary = 0b1010       # 2진수 (10)
octal = 0o17          # 8진수 (15)
hexadecimal = 0xFF    # 16진수 (255)
```

### b. 실수형(`float`)

실수는 IEEE 754 표준의 64비트 부동소수점으로 표현되며, 약 15-17자리의 정밀도를 제공한다:

```python
# 실수 리터럴
a = 3.14
b = -0.001
c = 2.5e3       # 2500.0 (지수 표기법)
d = 1.5e-3      # 0.0015

# 부동소수점의 정밀도 한계
x = 0.1 + 0.2
print(x)               # 0.30000000000000004
print(format(x, '.17f'))  # 0.30000000000000004

# 무한대와 NaN
positive_inf = float('inf')
negative_inf = float('-inf')
not_a_number = float('nan')
```

## 1.8.5 불리언 타입(`bool`)과 `None`

### a. 불리언 타입

파이썬의 불리언 타입은 `True`와 `False` 두 값을 가진다. 불리언은 `int`의 서브클래스이며, `True`는 `1`, `False`는 `0`의 값을 가진다:

```python
# 불리언 값
a = True
b = False

# 정수로 변환
print(int(True))    # 1
print(int(False))   # 0

# 논리 연산
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# 비교 연산
print(5 > 3)        # True
print(5 == 10)      # False
```

### b. `None` 타입

`None`은 파이썬에서 "값이 없음"을 나타내는 특별한 상수이다. 자세한 내용은 [1.4.1 None 객체](./1_4_builtin_objects.md#141-none-객체)를 참조하라.

```python
# None 값
a = None
print(a)          # None
print(type(a))    # <class 'NoneType'>

# 함수에서의 사용
def func():
    pass          # 반환값이 없으면 None 반환

result = func()
print(result)     # None
```

## 1.8.6 튜플 (`tuple`)

튜플은 불변(immutable) 시퀀스 타입으로, 한번 생성하면 내용을 변경할 수 없다.

### a. 튜플 생성과 액세스

```python
# 튜플 생성 방법
empty = ()
single = (1,)     # 요소가 하나인 경우 쉼표 필수
numbers = (1, 2, 3, 4, 5)
mixed = (1, "hello", True)

# 패킹과 언패킹
coordinates = 10, 20      # 튜플 패킹 (괄호 생략 가능)
x, y = coordinates        # 튜플 언패킹
print(x, y)               # 10 20

# 인덱싱과 슬라이싱
print(numbers[0])         # 1
print(numbers[-1])        # 5
print(numbers[1:3])       # (2, 3)
```

### b. 튜플과 가변 객체

튜플은 불변이지만, 튜플 내부에 포함된 가변 객체는 변경할 수 있다:

```python
# 가변 객체를 포함한 튜플
t = ([1, 2], 3)
t[0].append(3)      # 내부 리스트 수정 가능
print(t)            # ([1, 2, 3], 3)

# t[0] = [4, 5]     # TypeError: 튜플 항목 자체는 변경 불가
```

### c. 튜플의 활용

```python
# 다중 반환값
def get_coordinates():
    return 10, 20   # 튜플 반환

x, y = get_coordinates()
print(f"x: {x}, y: {y}")  # x: 10, y: 20

# 딕셔너리 키로 사용
locations = {(0, 0): "원점", (1, 0): "x축 1단위"}
print(locations[(0, 0)])  # "원점"
```

## 1.8.7 리스트 (`list`)

리스트는 가변(mutable) 시퀀스 타입으로, 순서가 있고 다양한 타입의 요소를 저장할 수 있다.

### a. 리스트 생성과 액세스

```python
# 리스트 생성
empty = []
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", True, [1, 2]]

# 인덱싱과 슬라이싱
print(numbers[0])         # 1
print(numbers[-1])        # 5
print(numbers[1:3])       # [2, 3]
```

### b. 리스트 수정

```python
# 요소 변경
numbers = [1, 2, 3, 4, 5]
numbers[0] = 10
print(numbers)            # [10, 2, 3, 4, 5]

# 요소 추가
numbers.append(6)         # 끝에 추가
numbers.insert(1, 15)     # 특정 위치에 삽입
numbers.extend([7, 8])    # 리스트 확장
print(numbers)            # [10, 15, 2, 3, 4, 5, 6, 7, 8]

# 요소 제거
numbers.remove(3)         # 값으로 제거 (첫 번째 일치 항목)
popped = numbers.pop()    # 마지막 요소 제거 및 반환
popped_index = numbers.pop(1)  # 특정 인덱스 요소 제거
print(numbers)            # [10, 2, 4, 5, 6, 7]
```

### c. 리스트 메서드와 연산

```python
# 리스트 연산
list1 = [1, 2, 3]
list2 = [4, 5]
combined = list1 + list2    # [1, 2, 3, 4, 5]
repeated = list1 * 2        # [1, 2, 3, 1, 2, 3]

# 정렬 및 역순
numbers = [3, 1, 4, 1, 5]
numbers.sort()              # 원본 정렬
print(numbers)              # [1, 1, 3, 4, 5]
numbers.reverse()           # 원본 역순
print(numbers)              # [5, 4, 3, 1, 1]

# 복사
original = [1, 2, [3, 4]]
shallow_copy = original.copy()  # 얕은 복사
import copy
deep_copy = copy.deepcopy(original)  # 깊은 복사
```

## 1.8.8 딕셔너리 (`dict`)

딕셔너리는 키-값 쌍을 저장하는 가변(mutable) 매핑 타입이다. 키는 해시 가능한 타입이어야 한다.

### a. 딕셔너리 생성과 액세스

```python
# 딕셔너리 생성
empty = {}
person = {"name": "Kim", "age": 30}
scores = dict(math=90, english=85)

# 요소 액세스
print(person["name"])      # Kim
print(scores.get("math"))  # 90
print(scores.get("history", "과목 없음"))  # "과목 없음" (기본값 제공)
```

### b. 딕셔너리 수정

```python
# 요소 추가/수정
person = {"name": "Kim", "age": 30}
person["email"] = "kim@example.com"  # 추가
person["age"] = 31                   # 수정
print(person)  # {'name': 'Kim', 'age': 31, 'email': 'kim@example.com'}

# 요소 제거
removed = person.pop("age")          # 키로 제거 및 값 반환
del person["email"]                  # del 키워드로 제거
print(person)  # {'name': 'Kim'}

# 모든 항목 지우기
person.clear()
print(person)  # {}
```

### c. 딕셔너리 메서드와 연산

```python
# 키, 값, 항목에 접근
person = {"name": "Kim", "age": 30}
print(list(person.keys()))      # ['name', 'age']
print(list(person.values()))    # ['Kim', 30]
print(list(person.items()))     # [('name', 'Kim'), ('age', 30)]

# 딕셔너리 병합 (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = dict1 | dict2          # {'a': 1, 'b': 3, 'c': 4} (중복키는 후자 우선)

# 딕셔너리 병합 (3.9 이전)
merged = {**dict1, **dict2}     # {'a': 1, 'b': 3, 'c': 4}
```

### d. 순서가 있는 딕셔너리 (`OrderedDict`)

파이썬 3.7부터는 기본 딕셔너리도 삽입 순서가 보존되지만, 명시적으로 순서를 다루고 싶을 때는 `OrderedDict`를 사용할 수 있다:

```python
from collections import OrderedDict

# 순서가 중요한 딕셔너리
scores = OrderedDict([
    ("math", 90),     # 첫 번째 항목
    ("english", 85),  # 두 번째 항목
    ("science", 88)   # 세 번째 항목
])

# 순서가 보장된 순회
for subject, score in scores.items():
    print(f"{subject}: {score}")

# 순서 변경 메서드 사용
scores.move_to_end("math")  # "math"를 맨 끝으로 이동
print(list(scores.items()))  # [('english', 85), ('science', 88), ('math', 90)]

# 일반 딕셔너리와 OrderedDict의 차이점
d1 = {"a": 1, "b": 2}
d2 = {"b": 2, "a": 1}
print(d1 == d2)  # True (일반 딕셔너리는 순서 무시)

od1 = OrderedDict([("a", 1), ("b", 2)])
od2 = OrderedDict([("b", 2), ("a", 1)])
print(od1 == od2)  # False (OrderedDict는 순서도 비교)
```

## 1.8.9 바이트 타입 (`bytes`, `bytearray`)

### a. `bytes` (불변 바이트 시퀀스)

```python
# bytes 생성
empty = bytes()
data = bytes([65, 66, 67])
text_bytes = b"ABC"
print(data)       # b'ABC'
print(data[0])    # 65

# 문자열과 변환
text = "안녕"
encoded = text.encode('utf-8')
print(encoded)              # b'\xec\x95\x88\xeb\x85\x95'
print(encoded.decode('utf-8'))  # '안녕'
```

### b. `bytearray` (가변 바이트 시퀀스)

```python
# bytearray 생성
empty = bytearray()
data = bytearray([65, 66, 67])
print(data)       # bytearray(b'ABC')

# 수정 가능
data[0] = 68
print(data)       # bytearray(b'DBC')
```

## 1.8.10 집합 (`set`)

집합은 중복이 없는 해시 가능한 객체의 컬렉션이다.

### a. 집합 생성과 수정

```python
# 집합 생성
empty = set()
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 2, 3, 3, 3])  # 중복 제거됨
print(numbers)  # {1, 2, 3}

# 요소 추가/제거
fruits.add("grape")
fruits.remove("banana")    # 없으면 KeyError
fruits.discard("melon")    # 없어도 에러 없음
print(fruits)  # {'apple', 'orange', 'grape'}
```

### b. 집합 연산

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# 집합 연산
print(set1 | set2)          # 합집합: {1, 2, 3, 4, 5, 6}
print(set1 & set2)          # 교집합: {3, 4}
print(set1 - set2)          # 차집합: {1, 2}
print(set1 ^ set2)          # 대칭차: {1, 2, 5, 6}

# 부분집합 관계
set3 = {3, 4}
print(set3.issubset(set1))  # True (set3 ⊆ set1)
print(set1.issuperset(set3)) # True (set1 ⊇ set3)
```

## 1.8.11 불변 타입인 `frozenset`

`frozenset`은 불변(immutable) 집합으로, 딕셔너리 키나 다른 집합의 요소로 사용 가능하다:

```python
# frozenset 생성
frozen = frozenset([1, 2, 3])
print(frozen)  # frozenset({1, 2, 3})

# 딕셔너리 키로 사용
groups = {frozenset([1, 2]): "그룹 A", frozenset([3, 4]): "그룹 B"}

# 일반 집합처럼 집합 연산 가능
frozen2 = frozenset([3, 4, 5])
print(frozen & frozen2)  # frozenset({3})
```

## 1.8.12 타입 힌트와 데이터 타입

파이썬의 타입 힌트 시스템에서 기본 데이터 타입을 활용하는 방법에 대한 자세한 내용은 [1.2 타입 시스템과 타입 힌트](./1_2_type_system.md)를 참조하라.

```python
# 기본 데이터 타입의 타입 힌트 (Python 3.9+)
def process_data(text: str, count: int, settings: dict[str, bool]) -> list[int]:
    result: list[int] = []
    # 처리 로직
    return result

# 3.9 이전 버전
from typing import Dict, List
def process_data_legacy(text: str, count: int, settings: Dict[str, bool]) -> List[int]:
    pass
```

> [목차로 돌아가기](../../README.md) | [이전: 함수형 프로그래밍 요소](./1_7_functional_programming.md)
