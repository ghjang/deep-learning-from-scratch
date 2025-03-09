# 1.1 파이썬 기본 개념과 주의사항

> [목차로 돌아가기](../README.md) | [다음: 파이썬 기본 자료형](./1_2_data_types.md)

## 파이썬 2와 3의 주요 차이점

__파이썬 '3' 버전은 '2' 버전과 호환되지 않는다.__ 즉 '하위 호환성'이 없다. 기본적인 언어 문법은 바뀌지 않았지만, 언어가 발전하면서 새로운 기능이 추가되거나 기존 기능이 변경되기도 했다.

* __'/' 연산자를 이용한 나눗셈의 결과가 '타입'이 달라졌다.__

  '2' 버전에서는 '정수 나누기 정수'의 결과가 '정수'로 나오지만, '3' 버전에서는 '실수'로 나온다. 예를 들어서 '2' 버전에서 '7 / 5'는 '1'이지만, '3' 버전에서는 '1.4'이다. 또한 '3' 버전에서 '정수 / 정수'의 결과가 '정수'이어도 결과 타입이 '실수'로 바뀌었다.

* __'int' 기본 타입의 숫자 표현 범위가 무한대로 바뀌었다.__

  '2' 버전에서는 '정수'를 표현하는 자료 타입으로 'int'와 'long'이 있었다. 'int'는 '32비트'나 '64비트'로 제한되어 있었고, 'long'은 '무한대'로 표현할 수 있었다. '3' 버전에서는 'int' 자료형만 있고, '메모리가 허용하는 한 무제한으로 큰 정수'를 표현할 수 있다.

## 타입 힌트

* __'타입 힌트' 기능을 모듈로 지원하던 것을 언어 자체에서 지원하는 것이 있다.__

   파이썬은 '동적 타이핑' 언어이다. 즉 '변수'의 '타입'을 명시적으로 지정하지 않아도 된다. 하지만 '타입 힌트'를 사용하면 '변수'의 '타입'을 명시적으로 지정할 수 있다. '타입 힌트'는 코드를 읽기 쉽게 만들어주고, '타입'을 잘못 사용하는 실수를 줄여준다. 또한 사용하는 IDE에서 타입 힌트를 보고 코드를 작성하는 데 도움을 준다. 예를 들어서 특정 코드 위에 마우스를 올리면 '타입 힌트'를 볼 수 있고, '자동 완성' 기능을 사용할 때도 도움이 된다.

  최신 버전의 파이썬 언어에서도 'typing' 모듈을 사용해서 '타입 힌트'를 여전히 지정할 수 있지만, 가능하면 파이썬 언어에 내장된 '타입 힌트'를 사용하는 것이 좋다.
  
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

  위 예제에서 볼 수 있듯이:
  * 내장 컬렉션 타입(list, tuple, dict 등)을 직접 타입 힌트로 사용
  * Optional[T]는 T | None으로 단순화
  * Union[T1, T2]는 T1 | T2로 단순화

* __긴 타입 이름에 대한 alias를 사용할 수 있다.__

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
      return sum(numbers)

  def process_config(config: StringDict) -> StringDict:
      return {k.upper(): v.upper() for k, v in config.items()}

  def process_data(data: MixedType) -> str:
      return str(data)
  ```

  '타입 힌트'에서 별칭을 사용하면 코드가 더 읽기 쉬워지고 타입 힌트를 재사용할 수 있다.
  특히 복잡한 제네릭 타입을 자주 사용할 때 유용하다.

> [목차로 돌아가기](../README.md) | [다음: 파이썬 기본 자료형](./1_2_data_types.md)
