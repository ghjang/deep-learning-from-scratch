# 1.5 타입 힌트 심화: 값과 타입의 경계

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 객체](./1_4_builtin_objects.md) | [다음: 데이터 분석 라이브러리](../2_modules_related_to_ds/2_1_data_analysis_libraries.md)

## 1.5.1 파이썬 타입 힌트의 이중성 이해하기

파이썬 타입 힌트 시스템에서는 '타입'과 '값'의 경계가 종종 모호해진다. 특히 `None`, `...`(Ellipsis), `True`, `False` 등의 싱글톤 객체들은 값이면서도 타입 힌트 컨텍스트에서 특별한 의미를 갖는다. 이러한 이중성은 파이썬의 동적 타입 시스템과 정적 타입 검사를 조화시키기 위한 설계 선택이지만, 처음 접하는 사람들에게는 혼란을 줄 수 있다.

### a. 값이자 타입 힌트로 사용되는 객체들

다음 객체들은 일반 코드에서는 값으로, 타입 힌트 컨텍스트에서는 특별한 타입 의미를 갖는다:

```python
# None: 값으로 사용
x = None
print(x)  # None

# None: 타입 힌트로 사용
def func() -> None:
    pass  # 반환값이 없음을 명시

# Ellipsis(...): 값으로 사용
placeholder = ...
print(placeholder)  # Ellipsis

# Ellipsis: 타입 힌트로 사용
from typing import Tuple
Vector = Tuple[int, ...]  # 가변 길이 튜플
```

## 1.5.2 타입 힌트에서 타입과 값의 구분

### a. Literal 타입을 통한 값의 타입화

`typing.Literal`은 구체적인 값 자체를 타입으로 승격시키는 방법을 제공한다:

```python
from typing import Literal

# 특정 값을 타입으로 제한
def align_text(alignment: Literal["left", "center", "right"]) -> str:
    return f"Text is {alignment} aligned"

# 값 제한 확인
align_text("left")     # 허용
align_text("center")   # 허용
# align_text("top")    # 타입 검사 도구에서 오류 (실행 시에는 오류 없음)
```

### b. None과 Optional의 관계

`None`은 값이지만 타입 힌트 맥락에서는 특별한 의미가 있다:

```python
from typing import Optional

# 아래 두 함수 선언은 동일한 의미
def legacy_func(x: Optional[int]) -> int:  # 3.9 이전
    return x or 0

def modern_func(x: int | None) -> int:     # 3.9 이후
    return x or 0
```

## 1.5.3 타입 연산자와 값 연산자의 구분

### a. `|` 연산자의 이중적 사용

파이썬 3.10 이후, `|` 연산자는 집합 연산과 타입 힌트에서 모두 사용된다:

```python
# 값 컨텍스트에서 | 연산자 (집합의 합집합)
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union = set1 | set2
print(union)  # {1, 2, 3, 4, 5}

# 타입 힌트 컨텍스트에서 | 연산자 (타입 유니온)
def process(x: int | str) -> None:
    print(x)
```

### b. 타입 정의에서 값의 사용

클래스와 같은 타입 자체와 그 클래스의 인스턴스인 값을 구분하는 것이 중요하다:

```python
class User:
    def __init__(self, name: str):
        self.name = name

# 사용 예시
def get_user_name(user: User) -> str:  # User는 타입으로 사용
    return user.name

# 값 컨텍스트
u = User("Alice")  # User는 클래스(생성자)로 사용, u는 값(인스턴스)
```

## 1.5.4 타입 힌트 시스템의 확장된 사용법

### a. TypeVar와 제네릭 함수

`TypeVar`를 사용한 제네릭 타입 정의:

```python
from typing import TypeVar, Sequence

T = TypeVar('T')  # 타입 변수 정의

def first_element(seq: Sequence[T]) -> T:
    return seq[0]

# 사용 예시
result1 = first_element([1, 2, 3])  # int로 추론
result2 = first_element(["a", "b"])  # str로 추론
```

### b. 타입으로서의 타입(Type[...])

메타프로그래밍과 팩토리 함수에서는 타입 그 자체도 타입 힌트로 사용될 수 있다:

```python
from typing import Type

class Base:
    pass

class Derived(Base):
    pass

def create_instance(cls: Type[Base]) -> Base:
    return cls()

# 사용 예시
b = create_instance(Base)     # 정상
d = create_instance(Derived)  # 정상 (Derived는 Base의 서브클래스)
```

## 1.5.5 타입 힌트와 실제 런타임 동작의 차이

타입 힌트는 코드 실행에 영향을 주지 않는다는 점을 기억하는 것이 중요하다:

```python
# 타입 힌트와 실제 동작이 다른 경우
def add(a: int, b: int) -> int:
    # 실제로는 문자열 연결도 수행 가능
    return a + b

print(add(1, 2))     # 3
print(add("a", "b")) # "ab" - 타입 힌트와 다르지만 실행됨

# 타입 검사 도구만 오류 감지
# mypy add.py - "error: Argument 1 to "add" has incompatible type "str"; expected "int""
```

파이썬의 타입 힌트 시스템은 정적 타입 언어의 장점을 동적 타입 언어에 도입하려는 시도지만, 두 패러다임의 경계에서 발생하는 혼란은 불가피하다. 타입과 값이 명확히 구분되는 정적 타입 언어와 달리, 파이썬에서는 이 둘 사이의 경계가 모호한 경우가 많다.

타입 힌트를 효과적으로 사용하려면 이러한 특성을 이해하고, 타입 체커(mypy 등)를 통해 문제를 조기에 발견하는 습관을 기르는 것이 중요하다.

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 객체](./1_4_builtin_objects.md) | [다음: 데이터 분석 라이브러리](../2_modules_related_to_ds/2_1_data_analysis_libraries.md)
