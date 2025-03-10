# 1.5 타입 힌트 심화: 값과 타입의 경계

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 객체](./1_4_builtin_objects.md) | [다음: 내장 전역 함수](./1_6_builtin_functions.md)

----
> **참고**: 이 문서는 파이썬 타입 힌트의 심화 내용을 다룬다. 기본적인 타입 힌트 개념과 사용법은 [1.2 타입 시스템과 타입 힌트](./1_2_type_system.md) 문서를 먼저 참조하기 바란다.
----

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

특히 문자열 리터럴 값에서는 `Literal` 타입이 필수적이다. 다음과 같이 직접 유니온으로 연결하는 것은 불가능하다:

```python
# 잘못된 시도 - 타입 시스템에서는 str | str | str로 해석됨 (즉, 단순히 str)
def wrong_align_text(alignment: "left" | "center" | "right") -> str:  
    return f"Text is {alignment} aligned"

# 이 경우 어떤 문자열이든 허용됨
wrong_align_text("top")       # 타입 체커도 허용 (의도와 다름)
wrong_align_text("anything")  # 타입 체커도 허용 (의도와 다름)
```

### b. 숫자 리터럴 타입

`Literal`은 숫자 값에도 동일하게 적용할 수 있다:

```python
from typing import Literal

# 특정 숫자 값만 허용하는 함수
def set_difficulty(level: Literal[1, 2, 3]) -> str:
    difficulties = {
        1: "쉬움",
        2: "보통",
        3: "어려움"
    }
    return f"난이도: {difficulties[level]}"

# 허용되는 값
set_difficulty(1)  # "난이도: 쉬움"
set_difficulty(3)  # "난이도: 어려움"

# 타입 체커 오류 (실행 시에는 문제 없음)
# set_difficulty(4)  # error: Argument 1 to "set_difficulty" has incompatible type "Literal[4]"; expected "Literal[1, 2, 3]"
```

### c. 숫자 범위 타입 표현의 한계

현재 파이썬 타입 시스템에서는 연속된 숫자 범위(예: 1에서 100까지의 정수)를 직접 표현하는 방법은 제공되지 않는다. 이런 경우 다음과 같은 방법을 사용할 수 있다:

#### 1. NewType을 사용한 런타임 검증

```python
from typing import NewType, Any, cast

# 양의 정수를 나타내는 타입 정의
PositiveInt = NewType('PositiveInt', int)

def ensure_positive(value: int) -> PositiveInt:
    if value <= 0:
        raise ValueError("Value must be positive")
    return cast(PositiveInt, value)  # 타입 캐스팅

# 사용 예시
def process_positive(num: PositiveInt) -> None:
    print(f"Processing positive number: {num}")

# 함수 호출 시 변환 함수 사용
process_positive(ensure_positive(5))  # 정상 동작
try:
    process_positive(ensure_positive(-5))  # 런타임 오류 발생
except ValueError as e:
    print(e)  # "Value must be positive"
```

#### 2. 실제 검사는 함수 내부에서 수행

```python
def process_range(value: int) -> None:
    """
    1에서 100 사이의 정수를 처리하는 함수
    
    Args:
        value: 1-100 사이의 정수
    """
    if not (1 <= value <= 100):
        raise ValueError("Value must be between 1 and 100")
        
    print(f"Processing value: {value}")

# 타입 체커는 모든 정수를 허용하지만, 런타임에 범위 검사
process_range(50)  # 정상 동작
try:
    process_range(200)  # 런타임 오류
except ValueError as e:
    print(e)  # "Value must be between 1 and 100"
```

파이썬의 타입 힌트 시스템은 정적 분석용으로, 값의 범위를 강제하는 기능은 제한적이다. 따라서 범위 제한이 필요한 경우에는 런타임 검사를 별도로 구현해야 한다.

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

`TypeVar`는 제네릭 프로그래밍을 위한 '타입 변수'를 정의한다:

```python
from typing import TypeVar, Sequence

T = TypeVar('T')  # 타입 변수 정의 - 타입 파라미터로 사용됨

def first_element(seq: Sequence[T]) -> T:
    return seq[0]

# 사용 예시
result1 = first_element([1, 2, 3])  # int로 추론
result2 = first_element(["a", "b"])  # str로 추론
```

### b. TypeVar와 type 키워드의 차이

`TypeVar`와 `type` 키워드는 혼동하기 쉽지만 완전히 다른 목적을 가진다:

```python
from typing import TypeVar, List

# 1. TypeVar: 제네릭 타입 파라미터 정의
T = TypeVar('T')  # T는 어떤 타입이든 될 수 있는 타입 변수

# 제네릭 함수 - T는 호출 시점에 구체적인 타입으로 결정됨
def first_of_list(items: List[T]) -> T:
    return items[0]

# 2. type 키워드: 타입 별칭(alias) 정의
type IntList = list[int]  # IntList는 단순히 list[int]의 다른 이름

# 별칭을 사용한 함수 - 항상 동일한 타입으로 고정됨
def sum_of_ints(numbers: IntList) -> int:
    return sum(numbers)

# 사용 시 차이점
ints = [1, 2, 3]
strs = ["a", "b", "c"]

first_int = first_of_list(ints)    # T가 int로 결정됨
first_str = first_of_list(strs)    # T가 str로 결정됨

sum_result = sum_of_ints(ints)     # 정상 작동
# sum_of_ints(strs)                # 타입 오류: strs는 IntList(list[int])가 아님
```

이러한 차이점을 요약하면:

- `TypeVar`: 여러 다른 타입을 대체할 수 있는 '타입 변수'를 만든다(다형성 지원)
- `type`: 기존 타입에 새로운 이름을 부여하는 '타입 별칭'을 만든다(가독성 향상)

### c. 타입으로서의 타입(Type[...])

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

파이썬의 타입 힌트 시스템은 정적 타입 언어의 장점을 동적 타입 언어에 도입하려는 시도이지만, 두 패러다임의 경계에서 발생하는 혼란은 불가피하다. 타입과 값이 명확히 구분되는 정적 타입 언어와 달리, 파이썬에서는 이 둘 사이의 경계가 모호한 경우가 많다.

타입 힌트를 효과적으로 사용하려면 이러한 특성을 이해하고, 타입 체커(mypy 등)를 통해 문제를 조기에 발견하는 습관을 기르는 것이 중요하다.

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 객체](./1_4_builtin_objects.md) | [다음: 내장 전역 함수](./1_6_builtin_functions.md)
