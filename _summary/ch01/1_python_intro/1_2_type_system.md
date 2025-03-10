# 1.2 타입 시스템과 타입 힌트

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 특유의 주요 문법](./1_1_wicked_syntaxes.md) | [다음: 객체 시스템과 객체 지향 프로그래밍 지원](./1_3_oop_system.md)

## 1.2.1 파이썬은 동적 타이핑 언어지만 타입 힌트를 지원한다

파이썬의 타입 시스템은 다음과 같은 특징을 가진다:
  
### a. 동적 타이핑

변수에 타입을 선언하지 않고 사용 가능하다:

```python
# 동적 타이핑의 유연성
x = 10          # x는 정수
x = "hello"     # x가 문자열로 변경됨 (다른 타입 할당 가능)
x = [1, 2, 3]   # x가 리스트로 변경됨
```
  
### b. 강력한 타입 시스템

타입 변환은 명시적으로 수행해야 한다:

```python
num_str = "123"
# num_str + 1  # TypeError 발생: 문자열과 정수 연산 불가능
num = int(num_str)  # 명시적 타입 변환 필요
result = num + 1    # 이제 연산 가능 (124)
```
  
### c. 타입 힌트

실행 시점에 강제되지 않는 타입 정보를 '타입 힌트'로 제공할 수 있다:

```python
def greet(name: str) -> str:
    return "Hello, " + name
    
# 힌트가 있어도 다른 타입 전달 가능 (실행 오류는 없음)
greet(123)  # "Hello, 123"으로 동작 (타입 검사 도구만 경고)
```

실행 시점에 강제되지 않는 다는 것의 의미는 '타입 힌트'가 개발시에는 정보를 제공해 IDE 등에 도움을 주어 조기에 오류를 발견할 수 있게 하지만, 런타임에는 타입 힌트가 무시되어 동적 타이핑의 유연성을 그대로 유지한다는 것이다.

## 1.2.2 동적 타이핑과 타입 힌트는 다른 목적으로 사용된다

__두 개념의 주요 차이점__은 다음과 같다:
  
| 특성 | 동적 타이핑 | 타입 힌트 |
|------|------------|----------|
| 타입 검사 시점 | 런타임(실행 시점) | 개발 시점 (정적 분석) |
| 타입 오류 시 | 실행 중 오류 발생 | 코드 실행 전 경고 가능 |
| 적용 방식 | 언어의 기본 동작 | 선택적 주석 |
| 강제성 | 타입 규칙 항상 적용 | 힌트일 뿐, 강제 안 함 |

## 1.2.3 '타입 힌트' 기능이 발전했다

파이썬 3.9부터 많은 내장 타입들이 타입 힌트로 직접 사용 가능해졌다:

```python
# 이전 방식 (3.9 이전)
from typing import List, Dict, Tuple, Optional, Union

def process_data(numbers: List[int],
                pairs: Tuple[str, int],
                config: Dict[str, str],
                name: Optional[str]) -> Union[List[float], None]:
    if name is None:
        return None
    return [float(n) for n in numbers]

# 새로운 방식 (3.9+ 권장)
def process_data(numbers: list[int],
                pairs: tuple[str, int],
                config: dict[str, str],
                name: str | None) -> list[float] | None:
    if name is None:
        return None
    return [float(n) for n in numbers]
```

위 예제를 통해서 알 수 있는 내용은 다음과 같다:

* 내장 컬렉션 타입(`list`, `tuple`, `dict` 등)을 직접 타입 힌트로 사용
* `Optional[T]`는 `T | None`으로 단순화
* `Union[T1, T2]`는 `T1 | T2`로 단순화

## 1.2.4 긴 타입 이름에 대한 alias를 사용할 수 있다

긴 타입 이름에 대한 별칭은 두 가지 방식으로 만들 수 있다:

```python
from typing import List, Dict, Union

# 1. 직접 할당 방식 (3.9 이전)
IntList = List[int]
StringDict = Dict[str, str]
MixedType = Union[int, str, float]

# 2. type 키워드 사용 방식 (3.9+)
type IntList = list[int]
type StringDict = dict[str, str]
type MixedType = int | str | float

# 사용 예시
def process_numbers(numbers: IntList) -> int:
    return sum(numbers)  # [1, 2, 3] → 6

def process_config(config: StringDict) -> StringDict:
    return {k.upper(): v.upper() for k, v in config.items()}  # {"a": "b"} → {"A": "B"}

def process_data(data: MixedType) -> str:
    return str(data)  # 123 → "123", "hello" → "hello"
```

'타입 힌트'에서 별칭을 사용하면 코드가 더 읽기 쉬워지고 타입 힌트를 재사용할 수 있다.
특히 복잡한 제네릭 타입을 자주 사용할 때 유용하다.

## 1.2.5 재귀적 타입 정의가 가능하다

파이썬에서는 자기 자신을 참조하는 재귀적 타입을 정의할 수 있다:

```python
# 3.9 이전: 문자열 리터럴로 타입 이름 참조
from typing import List, Dict, Union

# JSON 값은 기본형(str, int, ...) 또는 
# JSON 값의 리스트나 JSON 키-값 쌍의 딕셔너리일 수 있음
JSONValue = Union[str, int, float, bool, None, List['JSONValue'], Dict[str, 'JSONValue']]

# 3.10 이후: 직접 타입 이름 사용 가능
type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]

# 연결 리스트 노드 예제
class Node:
    def __init__(self, value: int, next: 'Node' = None):
        self.value = value
        self.next = next
        
# 파이썬 3.10 이후 타입 별칭 사용
type TreeNode = dict[str, int | list['TreeNode']]
  
# 사용 예시 - 계층적 구조 표현 가능
tree: TreeNode = {
    "value": 1,
    "children": [
        {"value": 2, "children": []},
        {"value": 3, "children": [{"value": 4, "children": []}]}
    ]
}
```

재귀적 타입은 트리, 그래프, 중첩된 데이터 구조 등 자기 참조적 구조를 표현할 때 매우 유용하다.

## 1.2.6 타입 힌트에서 '...'(Ellipsis)를 활용한다

파이썬의 타입 힌트에서 `...`는 다음과 같은 용도로 사용된다.
  
### a. 가변 길이 튜플 타입 정의

```python
from typing import Tuple

# ... 사용으로 임의 개수의 int 요소를 가진 튜플 표현
Vector = Tuple[int, ...]

def process_vector(v: Vector) -> int:
    return sum(v)  # (1, 2) → 3, (1, 2, 3, 4, 5) → 15
    
# 사용 예시
process_vector((1, 2))          # 유효
process_vector((1, 2, 3, 4, 5)) # 유효
```
  
### b. 타입 체킹에서 "구현 예정" 표시

```python
def calculate_something(x: int, y: int) -> ...:
    # 반환 타입이 아직 결정되지 않았거나 복잡한 경우 
    # ... 사용으로 "추후 정의될 예정" 표시
    return x + y
```

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 특유의 주요 문법](./1_1_wicked_syntaxes.md) | [다음: 객체 시스템과 객체 지향 프로그래밍 지원](./1_3_oop_system.md)
