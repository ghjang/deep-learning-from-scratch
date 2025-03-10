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

## 1.5.2 기본 타입 힌트 패턴

### a. None과 Optional의 관계

`None`은 값이지만 타입 힌트 맥락에서는 특별한 의미를 갖는다:

```python
from typing import Optional

# 아래 두 함수 선언은 동일한 의미
def legacy_func(x: Optional[int]) -> int:  # 3.9 이전
    return x or 0

def modern_func(x: int | None) -> int:     # 3.9 이후
    return x or 0
```

### b. `|` 연산자와 유니온 타입

파이썬 3.10 이후, `|` 연산자는 타입 힌트와 일반 코드 모두에서 사용된다:

```python
# 타입 힌트 컨텍스트에서 | 연산자 (타입 유니온)
def process(x: int | str) -> None:
    print(x)

# 이는 다음과 같다 (3.9 이전)
from typing import Union
def process_old(x: Union[int, str]) -> None:
    print(x)
    
# 값 컨텍스트에서 | 연산자 (집합의 합집합)
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union = set1 | set2
print(union)  # {1, 2, 3, 4, 5}
```

### c. 타입 정의에서 값의 사용

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

## 1.5.3 특수 타입 힌트 기법

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

## 1.5.4 타입 힌트와 실제 런타임 동작의 차이

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

## 1.5.5 고급 타입 힌트 기법

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

### b. type, Type, TypeVar의 구분: 비슷하지만 다른 개념들

파이썬의 타입 시스템에는 비슷한 이름을 가진 여러 개념이 있어 혼란스러울 수 있다. 다음에 설명하는 이러한 타입 관련 도구들은 파이썬의 정적 타입 검사 기능을 강화하지만, 유사한 이름으로 인해 처음에는 혼란스러울 수 있다:

1. `type` 내장 함수: 객체의 타입을 반환하는 함수

   ```python
   print(type(42))  # <class 'int'>
   print(type("hello"))  # <class 'str'>
   ```

2. `type` 키워드 (Python 3.10+): 타입 별칭을 정의하는 키워드

    ```python
    type IntList = list[int]  # 타입 별칭 정의
    ```

3. `typing.Type`: 클래스 자체를 타입 힌트로 사용할 때 필요한 제네릭

    ```python
    from typing import Type
    
    def factory(cls: Type[Base]): ...  # Base 또는 그 서브클래스를 받는 타입 힌트
    ```

4. `typing.TypeVar`: 제네릭 프로그래밍을 위한 타입 변수 생성기

    ```python
    from typing import TypeVar
    
    T = TypeVar('T')  # 타입 변수 T 정의
    ```

    `TypeVar('T')`의 문자열 인자에 관해:
    * 문자열 인자는 타입 변수의 이름을 지정한다.
    * 관례적으로 `T`, `U`, `V` 또는 `K`, `V` 등 짧은 대문자를 사용한다.
    * 이 문자열은 오류 메시지나 IDE의 타입 정보 표시에서 참조된다.
    * 문자열은 변수 이름과 동일하게 지정하는 것이 권장된다 (`T = TypeVar('T')`).
    * 문자열로 어떤 이름도 사용 가능하나, 알파벳 한 글자나 설명적인 이름이 관례적이다

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

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 객체](./1_4_builtin_objects.md) | [다음: 내장 전역 함수](./1_6_builtin_functions.md)
