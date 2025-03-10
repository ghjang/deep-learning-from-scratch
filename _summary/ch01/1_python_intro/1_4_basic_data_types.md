# 1.4 기본 데이터 타입

> [목차로 돌아가기](../../README.md) | [이전: 객체 시스템과 객체 지향 프로그래밍 지원](./1_3_oop_system.md)

## 1.4.1 문자열 (str)

a. __'문자열'은 '작은 따옴표'나 '큰 따옴표'로 감싸서 표현할 수 있다.__

  예를 들어 'print("Hello, World!")'는 'Hello, World!'를 출력한다. 문자열의 타입명은 'str'이다.

b. __'"""'은 '여러 줄의 문자열'을 표현할 수 있다.__

  예를 들어 '"""Hello, World!"""'는 'Hello, World!'를 표현한다. '여러 줄의 문자열'을 표현할 때 유용하다. 예를 들어서 다음과 같이 사용할 수 있다:

    ```python
    print("""
        Hello,
        World!
    """)
    ```

  분리된 줄 상에의 '첫번째 줄'의 앞쪽 '공백'은 기본적으로 최종 '문자열'에 포함된다. 최종 '문자열'의 시작 부분에 '공백'을 넣지 않으려면 다음과 같이 '\\, 백슬래시'를 여는 '"""' 다음에 넣어주면 된다:

    ```python
    print("""\
        Hello,
        World!
    """)
    ```

  출력 결과는 다음과 같다:

    ```txt
    Hello,
        World!
    ```

  '"""'는 'docstring'이라고 불리는 '문자열'을 표현할 때도 사용된다. 'docstring'은 '모듈', '클래스', '함수' 등의 '설명'을 담고 있는 '문자열'이다. 예를 들어서 다음과 같이 '코드 문서화'를 하는데 사용할 수 있다:

    ```python
    def my_function():
        """함수의 설명을 적는다."""
        pass
    ```

## 1.4.2 숫자 타입 (int, float)

* __'숫자'는 '정수'와 '실수'로 나눌 수 있다.__

  예를 들어 'a = 10'은 'a'라는 '정수'에 '10'을 저장한다. 'b = 2.5'은 'b'라는 '실수'에 '2.5'를 저장한다.
  
  파이썬의 숫자 타입은 다음과 같은 특징이 있다:
  * int (정수)
    * 파이썬 2에서는 int가 32비트나 64비트로 고정
    * 파이썬 3에서는 메모리가 허용하는 한 무제한으로 큰 정수를 표현 가능

    ```python
    # 매우 큰 정수도 처리 가능
    big_num = 123456789123456789123456789
    print(type(big_num))  # <class 'int'>
    print(big_num * 2)    # 246913578246913578246913578
    ```

  * float (실수)
    * C의 double과 같은 64비트 부동소수점 사용
    * 약 15-17 자릿수의 정밀도 제공

    ```python
    # 부동소수점의 정밀도 한계
    x = 0.1 + 0.2
    print(x)              # 0.30000000000000004
    print(format(x, '.17f'))  # 0.30000000000000004
    ```

## 1.4.3 튜플 (tuple)

* __'튜플'은 '괄호'로 감싸서 표현할 수 있다.__

  예를 들어 'a = (1, 2, 3)'은 'a'라는 '튜플'에 '1, 2, 3'을 저장한다:

    ```python
    a = (1, 2, 3)
    print(a)    # 출력: (1, 2, 3)
    ```

    '튜플'은 '리스트'와 비슷하지만 '불변(immutable)'이다. 즉, 한번 생성된 튜플의 요소는 변경할 수 없다:

    ```python
    a = (1, 2, 3)
    # a[0] = 5    # 오류 발생: 'tuple' object does not support item assignment
    ```

    튜플의 원소에 접근하는 방식은 리스트와 동일하게 '대괄호'와 '인덱스'를 사용한다:

    ```python
    a = (1, 2, 3, 4, 5)
    print(a[0])     # 출력: 1
    print(a[2])     # 출력: 3
    print(a[-1])    # 출력: 5 (음수 인덱스는 끝에서부터 접근)
    ```

    하지만 튜플의 요소가 가변(mutable) 객체라면 그 객체 자체는 변경할 수 있다:

    ```python
    a = ([1, 2], 3)
    a[0].append(3)    # 튜플 내부의 리스트는 변경 가능
    print(a)    # 출력: ([1, 2, 3], 3)
    ```

    튜플은 항목이 하나만 있을 때 콤마를 꼭 붙여야 한다:

    ```python
    a = (1,)    # 이것은 튜플입니다
    b = (1)     # 이것은 정수입니다
    print(type(a))    # 출력: <class 'tuple'>
    print(type(b))    # 출력: <class 'int'>
    ```

    튜플은 '패킹'과 '언패킹'을 통해 다중 할당이 가능하다:

    ```python
    # 패킹
    coordinates = (10, 20)
    
    # 언패킹
    x, y = coordinates
    print(x)    # 출력: 10
    print(y)    # 출력: 20
    
    # 다중 할당
    a, b = 1, 2
    print(a, b)    # 출력: 1 2
    
    # 값 교환
    a, b = b, a
    print(a, b)    # 출력: 2 1
    ```

    리스트와 마찬가지로 튜플도 타입 힌트를 사용할 수 있다:

    ```python
    # 이전 방식 (3.9 이전)
    from typing import Tuple
    def get_coordinates() -> Tuple[int, int]:
        return (10, 20)
    ```

    ```python
    # 새로운 방식 (3.9+ 권장)
    def get_coordinates() -> tuple[int, int]:
        return (10, 20)
    ```

## 1.4.4 리스트 (list)

* __'리스트'는 '대괄호'로 감싸서 표현할 수 있다.__

  예를 들어 'a = [1, 2, 3]'은 'a'라는 '리스트'에 '1, 2, 3'을 저장한다:

    ```python
    a = [1, 2, 3]
    print(a)    # 출력: [1, 2, 3]
    ```

    리스트는 '대괄호'를 사용한 문법이 마치 'C/C++'의 '선형 메모리' 배치를 보장하는 '배열'과 비슷하다. 하지만 파이썬의 리스트는 선형 메모리를 보장하지 않는다. 리스트는 '동적 배열'로 구현되어 있어서 '동적으로 크기가 조절'된다. 또한 '다양한 타입'의 요소를 저장할 수 있다:

    ```python
    a = [1, 2.0, 'hello']
    print(a)    # 출력: [1, 2.0, 'hello']
    ```

    파이썬 언어 자체에는 '선형 메모리'를 보장하는 '배열'이 없다. '배열'을 사용하려면 'numpy' 라이브러리를 사용해야 한다.

    이런 리스트 표현에 타입 힌트를 사용할 수 있다:

    ```python
    # 이전 방식 (3.9 이전)
    from typing import List
    def process_data(numbers: List[int]):
        pass
    ```
  
    ```python
    # 새로운 방식 (3.9+ 권장)
    def process_data(numbers: list[int]):
        pass
    ```

## 1.4.5 딕셔너리 (dict)

* __'딕셔너리'는 '중괄호'로 감싸서 표현할 수 있다.__

  예를 들어 'b = {"apple": 100, "banana": 100}'은 'b'라는 '딕셔너리'에 'apple: 100'과 'banana: 100'을 저장한다:

    ```python
    b = {"apple": 100, "banana": 100}
    print(b)    # 출력: {'apple': 100, 'banana': 100}
    ```

    '딕셔너리'는 '키-값 쌍'을 저장하는 자료형이다. '키'는 '해시 가능한' 타입이어야 한다. '값'은 '어떤 타입'이든 상관 없다. '딕셔너리'는 '해시 테이블'로 구현되어 있어서 '키'를 이용해서 '값'을 빠르게 찾을 수 있다.

    이런 딕셔너리 표현에 타입 힌트를 사용할 수 있다:

    ```python
    # 이전 방식 (3.9 이전)
    from typing import Dict
    def process_data(config: Dict[str, str]):
        pass
    ```
  
    ```python
    # 새로운 방식 (3.9+ 권장)
    def process_data(config: dict[str, str]):
        pass
    ```

    여기서 'dict[K, V]'는 다음과 같은 의미를 가진다:

  * K: 딕셔너리의 '키'의 타입
  * V: 딕셔너리의 '값'의 타입

  예를 들어:

    ```python
    from typing import Any  # Any, TypeVar 등의 특수 타입은 여전히 typing 모듈에서 import 필요
    
    scores: dict[str, int] = {"math": 90, "english": 85}    # 키는 문자열, 값은 정수
    mixed: dict[str, Any] = {"name": "Bob", "age": 20}      # 키는 문자열, 값은 아무 타입
    nested: dict[str, dict[str, int]] = {                   # 중첩된 딕셔너리
        "semester1": {"math": 90, "english": 85},
        "semester2": {"math": 88, "english": 87}
    }
    ```

    다음은 키가 '문자열'이 아닌 '해쉬 가능한' 타입인 딕셔너리의 예시이다:

    ```python
    # 키가 정수인 딕셔너리
    counts: dict[int, str] = {1: "one", 2: "two", 3: "three"}
    
    # 키가 튜플인 딕셔너리
    coords: dict[tuple[int, int], str] = {
        (0, 0): "origin",
        (1, 0): "right",
        (0, 1): "up"
    }
    ```

    파이썬 3.7부터는 기본 딕셔너리도 '삽입 순서'를 보장하지만,
    명시적으로 '순서가 있는 딕셔너리'가 필요할 때는 'collections.OrderedDict'를 사용할 수 있다:

    ```python
    from collections import OrderedDict
    from typing import OrderedDict as OrderedDictType  # 3.9 이전 타입 힌트용

    # 3.9 이전
    def process_ordered(config: OrderedDictType[str, int]):
        pass

    # 3.9 이후
    def process_ordered(config: OrderedDict[str, int]):
        pass

    # 사용 예시
    scores = OrderedDict([
        ("math", 90),     # 첫 번째 항목
        ("english", 85),  # 두 번째 항목
        ("science", 88)   # 세 번째 항목
    ])
    
    # 순서가 보장된 순회
    for subject, score in scores.items():
        print(f"{subject}: {score}")
    ```

> [목차로 돌아가기](../../README.md) | [이전: 객체 시스템과 객체 지향 프로그래밍 지원](./1_3_oop_system.md)
