# 1. 파이썬 인트로

'1장 헬로 파이썬'을 읽고 기억해야할만 한 내용들을 정리한다.

## 1.1 주의사항

__파이썬 '3' 버전은 '2' 버전과 호환되지 않는다.__ 즉 '하위 호환성'이 없다. 기본적인 언어 문법은 바뀌지 않았지만, 언어가 발전하면서 새로운 기능이 추가되거나 기존 기능이 변경되기도 했다.

* __'/' 연산자를 이용한 나눗셈의 결과가 '타입'이 달라졌다.__

  '2' 버전에서는 '정수 나누기 정수'의 결과가 '정수'로 나오지만, '3' 버전에서는 '실수'로 나온다. 예를 들어서 '2' 버전에서 '7 / 5'는 '1'이지만, '3' 버전에서는 '1.4'이다. 또한 '3' 버전에서 '정수 / 정수'의 결과가 '정수'이어도 결과 타입이 '실수'로 바뀌었다.

* __'int' 기본 타입의 숫자 표현 범위가 무한대로 바뀌었다.__

  '2' 버전에서는 '정수'를 표현하는 자료 타입으로 'int'와 'long'이 있었다. 'int'는 '32비트'나 '64비트'로 제한되어 있었고, 'long'은 '무한대'로 표현할 수 있었다. '3' 버전에서는 'int' 자료형만 있고, '메모리가 허용하는 한 무제한으로 큰 정수'를 표현할 수 있다.

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

## 1.2 파이썬 기본 자료형

* __'문자열'은 '작은 따옴표'나 '큰 따옴표'로 감싸서 표현할 수 있다.__

  예를 들어 'print("Hello, World!")'는 'Hello, World!'를 출력한다. 문자열의 타입명은 'str'이다.

* __'"""'은 '여러 줄의 문자열'을 표현할 수 있다.__

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

## 1.3 파이썬 특이 기본 문법

### 1.3.1 가변(Mutable)과 불변(Immutable) 타입

* __파이썬의 데이터 타입은 '가변(mutable)'과 '불변(immutable)'으로 나뉜다.__

  이 구분은 객체가 생성된 후 그 내용을 변경할 수 있는지 여부를 결정한다:

  * 불변(Immutable) 타입: 한번 생성되면 내용을 변경할 수 없는 객체
    * int, float, bool, str, tuple, frozenset, bytes
  
  * 가변(Mutable) 타입: 내용을 자유롭게 변경할 수 있는 객체
    * list, dict, set, bytearray, 사용자 정의 클래스
  
  불변 객체는 내용을 변경하는 연산을 수행하면 새로운 객체가 생성된다:

  ```python
  # 문자열(불변 객체) 연산
  s = "hello"
  id_before = id(s)  # 객체의 메모리 주소
  
  s = s + " world"   # 새로운 객체 생성
  id_after = id(s)
  
  print(id_before == id_after)  # False - 다른 객체임
  ```

  반면 가변 객체는 내용 변경 시 같은 객체를 유지한다:

  ```python
  # 리스트(가변 객체) 연산
  lst = [1, 2, 3]
  id_before = id(lst)
  
  lst.append(4)  # 같은 객체의 내용 수정
  id_after = id(lst)
  
  print(id_before == id_after)  # True - 동일한 객체임
  ```

* __가변성이 함수 호출과 변수 할당에 미치는 영향__

  불변 객체는 함수에 전달되거나 다른 변수에 할당될 때 전체가 복사된다(값 전달):
  
  ```python
  def modify_value(x):
      x = x + 1  # 새 객체 생성
      return x
  
  num = 10
  result = modify_value(num)
  print(num)     # 10 (원본 변경 없음)
  print(result)  # 11
  ```
  
  가변 객체는 함수에 전달되거나 다른 변수에 할당될 때 참조가 전달된다(참조 전달):
  
  ```python
  def modify_list(lst):
      lst.append(4)  # 원본 객체 수정
  
  numbers = [1, 2, 3]
  modify_list(numbers)
  print(numbers)  # [1, 2, 3, 4] (원본이 변경됨)
  ```

* __불변성(immutability)의 장점__

  1. 스레드 안전성: 여러 스레드가 동일한 객체에 접근해도 값이 변하지 않음
  2. 예측 가능성: 코드의 다른 부분에서 객체를 변경하지 않을 것이라는 보장
  3. 해시 가능: 딕셔너리 키나 집합의 요소로 사용 가능
  
* __가변성(mutability)의 장점__

  1. 효율성: 큰 데이터의 일부만 수정할 때 전체를 복사할 필요 없음
  2. 메모리 효율: 수정이 필요할 때마다 새 객체를 생성하지 않음
  3. 알고리즘 구현: 특정 알고리즘(정렬, 검색 등)을 더 직관적으로 구현 가능

### 1.3.2 시퀀스 타입과 슬라이싱

* __파이썬의 '시퀀스 타입'은 순서가 있는 데이터 컬렉션이다.__

  파이썬에는 다음과 같은 주요 시퀀스 타입들이 있다:
  
  * 문자열(str): 문자들의 시퀀스
  * 리스트(list): 변경 가능한(mutable) 객체들의 시퀀스
  * 튜플(tuple): 변경 불가능한(immutable) 객체들의 시퀀스
  * 범위(range): 정수 시퀀스(연속적인 숫자들)
  * 바이트(bytes): 바이트의 불변 시퀀스
  * 바이트배열(bytearray): 바이트의 가변 시퀀스
  
  모든 시퀀스 타입은 다음과 같은 공통 특징을 가진다:
  
  * 인덱싱을 통한 접근 가능 (0부터 시작)
  * 슬라이싱 연산 지원
  * len() 함수를 통한 길이 확인 가능
  * 반복문(for)에서 순회 가능
  * * 연산자(concatenation)와 * 연산자(repetition) 지원
  * in 연산자를 통한 포함 여부 확인

* __'슬라이싱'은 시퀀스의 일부분을 추출하는 강력한 기능이다.__

  모든 시퀀스 타입에서 공통적으로 사용할 수 있는 슬라이싱 문법은 다음과 같다:

  ```python
  sequence[start:stop:step]
  ```

  각 매개변수의 의미:
  * start: 시작 인덱스 (포함)
  * stop: 종료 인덱스 (미포함)
  * step: 인덱스 증가량

  슬라이싱의 주요 특징:
  
  ```python
  # 기본 슬라이싱 [시작:끝]
  a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  print(a[2:5])     # 출력: [2, 3, 4] (인덱스 2부터 5 전까지)
  
  # 시작 인덱스 생략 (처음부터)
  print(a[:5])      # 출력: [0, 1, 2, 3, 4]
  
  # 끝 인덱스 생략 (끝까지)
  print(a[5:])      # 출력: [5, 6, 7, 8, 9]
  
  # 스텝 사용 [시작:끝:스텝]
  print(a[1:9:2])   # 출력: [1, 3, 5, 7] (1부터 9 전까지 2 간격으로)
  
  # 음수 인덱스 사용
  print(a[-5:-2])   # 출력: [5, 6, 7] (끝에서 5번째부터 끝에서 2번째 전까지)
  
  # 전체 복사
  b = a[:]          # 리스트의 얕은 복사(shallow copy)
  
  # 리스트 뒤집기
  print(a[::-1])    # 출력: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  ```

* __다양한 시퀀스 타입에 적용되는 슬라이싱:__

  ```python
  # 문자열 슬라이싱
  s = "Hello Python"
  print(s[6:])      # 출력: "Python"
  print(s[:5])      # 출력: "Hello"
  print(s[::-1])    # 출력: "nohtyP olleH"
  
  # 튜플 슬라이싱
  t = (0, 1, 2, 3, 4, 5)
  print(t[1:5:2])   # 출력: (1, 3)
  
  # range 객체의 슬라이싱 (결과는 다시 range 객체가 됨)
  r = range(10)
  print(list(r[2:8:2]))  # 출력: [2, 4, 6]
  ```

* __슬라이싱은 원본 변경 없이 새로운 객체를 반환한다.__

  원본 시퀀스는 변경되지 않고, 슬라이싱 결과로 원본과 동일한 타입의 새 객체가 생성된다.
  단, 얕은 복사(shallow copy)이므로 리스트나 튜플의 요소가 가변 객체인 경우 요소 자체는 공유된다.
  
  '얕은 복사(shallow copy)'란 최상위 컨테이너는 새로 생성하지만, 내부 요소는 원본 객체와 동일한 참조를 가리키는 것을 말한다:
  
  ```python
  # 얕은 복사 예시
  original = [1, 2, [3, 4]]
  shallow_copy = original[:]  # 슬라이싱으로 얕은 복사
  
  # 최상위 컨테이너는 다른 객체다
  print(original is shallow_copy)  # False
  
  # 내부 요소 변경 시 영향 관계 확인
  shallow_copy[0] = 99  # 불변 객체 변경
  print(original)       # [1, 2, [3, 4]] - 원본에 영향 없음
  
  shallow_copy[2][0] = 33  # 가변 객체(내부 리스트)의 요소 변경
  print(original)          # [1, 2, [33, 4]] - 원본도 함께 변경됨
  ```
  
  반면 '깊은 복사(deep copy)'는 객체 내부의 모든 요소까지 재귀적으로 복사하는 것을 의미하며, 이를 위해서는 `copy` 모듈의 `deepcopy` 함수를 사용한다:
  
  ```python
  import copy
  
  original = [1, 2, [3, 4]]
  deep_copy = copy.deepcopy(original)
  
  deep_copy[2][0] = 33  # 가변 객체(내부 리스트)의 요소 변경
  print(original)       # [1, 2, [3, 4]] - 원본에 영향 없음
  print(deep_copy)      # [1, 2, [33, 4]]
  ```

* __시퀀스 타입과 이터러블(Iterable)의 관계__

  파이썬에서 시퀀스 타입은 모두 '이터러블(Iterable)'의 특성을 가진다. 이터러블이란 하나씩 차례대로 꺼내어 쓸 수 있는 객체를 의미한다. 시퀀스 타입 외에도 여러 이터러블이 존재한다:
  
  * 시퀀스 타입 - 리스트, 튜플, 문자열, range 등
  * 집합 타입 - set, frozenset
  * 사전 타입 - dict
  * 제너레이터(generator)
  * 파일 객체
  * 사용자 정의 이터러블(`__iter__()` 메서드를 구현한 객체)
  
  이터러블의 핵심 특징:
  
  1. `for` 루프에서 순회 가능:

     ```python
     # 여러 이터러블을 순회하는 예
     for item in [1, 2, 3]:  # 리스트
         print(item)
     
     for char in "Python":   # 문자열
         print(char)
     
     for key in {"a": 1, "b": 2}:  # 딕셔너리(키가 순회됨)
         print(key)
     ```
  
  2. 반복자(Iterator) 생성 가능:

     ```python
     # 이터러블로부터 반복자 생성
     my_list = [1, 2, 3]
     iterator = iter(my_list)  # __iter__() 메서드 호출
     
     # 반복자에서 값 가져오기
     print(next(iterator))  # 1
     print(next(iterator))  # 2
     print(next(iterator))  # 3
     # print(next(iterator))  # StopIteration 예외 발생
     ```
  
  3. 컴프리헨션(comprehension)과 함께 사용 가능:

     ```python
     # 리스트 컴프리헨션
     squares = [x**2 for x in range(5)]
     print(squares)  # [0, 1, 4, 9, 16]
     
     # 딕셔너리 컴프리헨션
     word = "hello"
     char_positions = {char: idx for idx, char in enumerate(word)}
     print(char_positions)  # {'h': 0, 'e': 1, 'l': 3, 'o': 4}
     ```
  
  4. 내장 함수와 함께 사용 가능:

     ```python
     print(sum([1, 2, 3, 4]))  # 10
     print(max("python"))      # 'y'
     print(min({5, 3, 8, 1}))  # 1
     print(sorted("hello"))    # ['e', 'h', 'l', 'l', 'o']
     ```

### 1.3.3 객체 지향 특성

* __파이썬에서 모든 표현 대상은 '객체'이다.__

  '객체'는 '메모리에 저장된 데이터'와 '데이터를 처리하는 함수'를 가지고 있다. '객체'는 '변수'에 할당할 수 있다. '변수'는 '객체'를 가리키는 '레퍼런스'를 가지고 있다.

  'class'로 명시적으로 표현되는 타입뿐만 아니라 'int', 'float', 'list', 'dict' 등의 내장 타입도 '객체'이다. 다음과 같은 코드는 모두 '객체'를 생성한다:

    ```python
    a = 10
    b = 2.5
    c = [1, 2, 3]
    d = {"apple": 100, "banana": 100}
    ```

    단순 숫자값도 객체이기 때문에 다음과 같이 메소드를 호출할 수 있다:

    ```python
    print((10).bit_length())    # 출력: 4
    ```

* __파이썬에서 모든 표현 대상은 'object' 클래스를 상속받은 '객체'이다.__

  파이썬에서는 모든 타입이 'object' 클래스를 암묵적으로 상속받는다. 이는 모든 객체가 공통으로 가지는 기본 메서드와 속성이 있다는 것을 의미한다:

  ```python
  # 모든 타입은 object의 자식이다
  print(isinstance(42, object))          # True
  print(isinstance("hello", object))     # True
  print(isinstance([1, 2, 3], object))   # True
  print(isinstance(len, object))         # True (함수도 객체다)
  print(isinstance(type, object))        # True (타입도 객체다)
  
  # object 클래스에서 상속받은 공통 메서드들
  num = 42
  print(dir(num))  # 모든 속성과 메서드 목록 출력
  print(num.__class__)  # <class 'int'>
  print(num.__str__())  # "42"
  print(num.__repr__()) # "42"
  
  # 사용자 정의 클래스도 자동으로 object를 상속
  class MyClass:  # 암묵적으로 object 상속
      pass
      
  class MyExplicitClass(object):  # 명시적으로 object 상속 (동일한 의미)
      pass
  
  obj = MyClass()
  print(isinstance(obj, object))  # True
  print(obj.__class__.__bases__) # (<class 'object'>,)
  ```

  모든 객체가 공통적으로 갖는 주요 메서드들:
  * `__str__()`: 문자열 표현 (str() 함수가 호출)
  * `__repr__()`: 개발자를 위한 상세 문자열 표현
  * `__class__`: 객체의 타입 정보
  * `__doc__`: 문서화 문자열
  * `__dict__`: 객체의 속성 목록 (네임스페이스)
  * `__hash__()`: 해시 값 계산 (딕셔너리 키로 사용 가능한지 결정)
  * `__eq__()`: 동등성 비교 (== 연산자)

### 1.3.4 특수 연산자

* __'**' 연산자는 '거듭제곱'을 나타낸다.__

  예를 들어 '2 ** 3'은 '2의 3제곱'을 의미한다.

* __'//' 연산자는 '나눗셈의 몫'을 나타낸다.__

  예를 들어 '7 // 5'는 '7을 5로 나눈 몫'을 의미한다.

* __'is' 연산자는 '객체 식별자 비교'를 수행한다.__

  'is' 연산자는 두 변수가 메모리상에서 동일한 객체를 참조하는지 비교한다.
  
  * '==' 연산자: 논리적 동등성(logical equality) 비교 - 두 객체의 값이 같은지 비교
  * 'is' 연산자: 물리적 동일성(physical identity) 비교 - 두 변수가 실제로 메모리 상 같은 객체를 가리키는지 비교

  이는 id() 함수가 반환하는 객체의 고유 식별자(메모리 주소)가 동일한지 비교하는 것과 같다:

  ```python
  # == 연산자 vs is 연산자
  a = [1, 2, 3]
  b = [1, 2, 3]
  c = a
  
  print(a == b)  # True (값이 동등함 - 내용이 같음)
  print(a is b)  # False (다른 객체 - 다른 메모리 주소)
  print(a is c)  # True (같은 객체를 참조 - 동일한 메모리 주소)
  
  # id() 함수로 객체 메모리 주소 확인
  print(id(a))   # 예: 140233683913024
  print(id(b))   # 예: 140233683913344 (다름)
  print(id(c))   # 예: 140233683913024 (a와 동일)
  
  # == 연산자는 값의 비교를 위해 __eq__ 메서드를 호출
  # is 연산자는 메모리 참조 비교로 더 빠르게 동작
  ```

  특히 None 값과의 비교는 항상 'is' 연산자를 사용해야 한다:

  ```python
  value = None
  
  # 권장 방식
  if value is None:
      print("값이 None입니다")
  
  # 권장하지 않음
  if value == None:
      print("== 연산자로 비교")
  ```

  파이썬에서는 일부 작은 정수와 같은 불변 객체들이 최적화를 위해 같은 객체로 관리되기도 한다:

  ```python
  # 작은 정수는 같은 객체로 관리됨 (-5부터 256까지)
  x = 5
  y = 5
  print(x is y)  # True (동일한 객체)
  
  # 큰 정수는 별도 객체로 관리됨
  large_x = 1000
  large_y = 1000
  print(large_x is large_y)  # 구현에 따라 다를 수 있음
  ```

  항상 값 비교에는 '==' 연산자를, 객체 동일성 비교에는 'is' 연산자를 사용하는 것이 권장된다.

### 1.3.5 None과 NoneType

* __'None'은 파이썬에서 '값이 없음'을 나타내는 특별한 객체이다.__

  None은 파이썬의 싱글톤(singleton) 객체로, 시스템 전체에 단 하나만 존재한다. 값의 부재, 초기화되지 않은 변수, 또는 함수에서 명시적인 반환값이 없을 때 사용된다.
  
  None의 주요 특징:
  
  1. NoneType이라는 고유한 타입을 가진다:
  
     ```python
     print(type(None))  # <class 'NoneType'>
     ```
  
  2. 메모리에 하나만 존재하는 싱글톤 객체이다:
  
     ```python
     a = None
     b = None
     print(a is b)  # True - 항상 같은 객체를 참조
     ```
  
  3. 불리언 컨텍스트에서 False로 평가된다:
  
     ```python
     print(bool(None))  # False
     
     if None:
         print("실행되지 않음")
     else:
         print("None은 False로 평가됨")
     ```
  
  4. 기본 반환값으로 사용된다:
  
     ```python
     def func_without_return():
         pass
         
     result = func_without_return()
     print(result)  # None
     print(result is None)  # True
     ```
  
  5. 변수나 객체의 초기화에 사용된다:
  
     ```python
     # 객체가 아직 존재하지 않음을 나타내기
     user = None
     
     # 나중에 초기화
     if condition:
         user = User("John")
     ```
  
  6. Optional 타입과 함께 자주 사용된다:
  
     ```python
     # 3.9 이전
     from typing import Optional
     
     def get_user(user_id: int) -> Optional[User]:
         if user_exists(user_id):
             return User(user_id)
         return None
         
     # 3.9 이후
     def get_user(user_id: int) -> User | None:
         if user_exists(user_id):
             return User(user_id)
         return None
     ```
  
  None은 '빈 값'을 나타내는 다른 객체들(빈 문자열 "", 빈 리스트 [], 숫자 0)과는 다르다. None은 값 자체가 없음을 의미한다:
  
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
  
  None 값을 비교할 때는 항상 'is' 연산자를 사용해야 한다:
  
  ```python
  # 권장 방식
  if value is None:
      print("값이 None입니다")
  
  if value is not None:
      print("값이 None이 아닙니다")
  ```

### 1.3.6 문법적 특이사항

* __'...'은 'pass'를 의미한다.__

  'pass'는 '아무것도 하지 않는 문장'이다. 예를 들어서 'if' 문에서 '아무것도 하지 않을 때' 사용할 수 있다:

    ```python
    if x < 0:
        print('negative')
    else:
        ... # 올바른 파이썬 문법이다.
    ```

## 1.4 파이썬 내장 전역 함수

* __'type()' 함수를 이용하면 주어진 '객체'의 '타입'을 알 수 있다.__

  예를 들어 'type(10)'은 'int'를 반환한다.

* __'isinstance()' 함수를 이용해서 주어진 '객체'가 '특정 타입'인지 확인할 수 있다.__

  예를 들어 'isinstance(10, int)'는 'True'를 반환한다.

* __'id()' 함수를 이용해서 '객체'의 '메모리 주소'를 알 수 있다.__

  예를 들어 'id(10)'은 '객체 10'의 '메모리 주소'를 반환한다.

* __'dir()' 함수를 이용해서 '객체'의 '속성'과 '메서드'를 알 수 있다.__

  예를 들어 'dir(10)'은 'int' 타입의 '객체'가 가지고 있는 '속성'과 '메서드'를 반환한다.

* __'len()' 함수를 이용해서 '리스트'나 '문자열'의 길이를 알 수 있다.__

  예를 들어 'len([1, 2, 3])'은 '3'을 반환한다.

* __'str()' 함수를 이용해서 '객체'를 '문자열'로 변환할 수 있다.__

  예를 들어 'str(10)'은 '10'을 반환한다.

* __'print()' 함수를 이용해서 '객체'를 표준 출력으로 출력할 수 있다.__

  예를 들어 'print(10)'은 '숫자 10'을 '문자열 10'으로 변환해서 표준 출력 장치로 출력한다.

* __'repr()' 함수를 이용해서 '객체'를 '개발자를 위한 문자열'로 변환할 수 있다.__

  예를 들어 'repr(10)'은 '10'을 반환한다. 'str()' 함수와 다르게 'repr()' 함수는 '개발자를 위한 문자열'을 반환한다.

* __'str()', 'print()', 'repr()' 함수들은 각자의 특별한 목적이 있다:__

  1. str(): 사람이 읽기 위한 문자열 변환
     * 일반 사용자를 위해 객체를 보기 좋게 표현
     * 객체의 `__str__()` 메서드 호출

     ```python
     from datetime import datetime
     
     date = datetime.now()
     print(str(date))  # 2024-01-20 15:30:00
     ```

  2. repr(): 개발자가 보기 위한 문자열 변환
     * 객체를 명확하게 구분할 수 있는 상세 정보 포함
     * 가능하면 객체를 재생성할 수 있는 문자열 형태로 반환
     * 객체의 `__repr__()` 메서드 호출
     * Python 셸에서 객체를 출력할 때 사용

     ```python
     date = datetime.now()
     print(repr(date))  # datetime.datetime(2024, 1, 20, 15, 30, 0, 123456)
     ```

  3. print(): 화면 출력 전용 함수
     * 객체를 str()로 변환한 후 화면에 출력
     * 자동으로 줄바꿈 추가 (end='\n' 기본값)
     * sep와 end 인자로 출력 형식 조정 가능

     ```python
     print(1, 2, 3, sep=',')  # 1,2,3
     print('hello', end='!')   # hello! (줄바꿈 없음)
     ```

  실제 사용 예시로 비교해보면:

  ```python
  # 문자열에서의 차이
  text = 'Hello\nWorld'
  print(str(text))   # Hello
                     # World     (줄바꿈 실행됨)
  print(repr(text))  # 'Hello\nWorld'  (이스케이프 문자가 보임)
  print(text)        # Hello
                     # World     (str()처럼 줄바꿈 실행)

  # 컬렉션에서의 차이
  data = [1, 'hello', 3.14]
  print(str(data))   # [1, 'hello', 3.14]
  print(repr(data))  # [1, 'hello', 3.14]  (이 경우 동일)
  
  # 사용자 정의 클래스에서의 차이
  class Person:
      def __init__(self, name, age):
          self.name = name
          self.age = age
      
      def __str__(self):
          return f"{self.name}, {self.age}세"
      
      def __repr__(self):
          return f"Person(name='{self.name}', age={self.age})"

  person = Person('홍길동', 20)
  print(str(person))   # 홍길동, 20세
  print(repr(person))  # Person(name='홍길동', age=20)
  print(person)        # 홍길동, 20세
  ```

* __'bool()' 함수와 파이썬의 '암묵적 형변환' 규칙:__

  bool() 함수는 객체의 진리값을 판단한다. 파이썬은 다음과 같은 명확한 규칙으로 객체를 bool 타입으로 변환한다:

  1. 숫자 타입:
     * False: 0, 0.0
     * True: 0이 아닌 모든 숫자

     ```python
     print(bool(0), bool(0.0))          # False False
     print(bool(1), bool(-1), bool(0.1)) # True True True
     ```

  2. 시퀀스/컬렉션 타입:
     * False: 비어있는 시퀀스/컬렉션 (길이가 0)
     * True: 비어있지 않은 시퀀스/컬렉션

     ```python
     print(bool([]), bool(""), bool({}), bool(set()))  # False False False False
     print(bool([1]), bool("a"), bool({1: 2}))        # True True True
     ```

  3. 특수 객체:
     * False: None
     * True: 그 외의 대부분의 객체

     ```python
     print(bool(None))                  # False
     print(bool(object()), bool(True))  # True True
     ```

  4. 사용자 정의 객체:
     * `__bool__()` 또는 `__len__()` 메서드로 진리값 결정

     ```python
     class MyClass:
         def __init__(self, value):
             self.value = value
         def __bool__(self):
             return self.value > 0

     obj = MyClass(1)
     print(bool(obj))  # True
     ```

  이러한 규칙은 if 문이나 while 문의 조건식에서도 동일하게 적용된다:

  ```python
  # 빈 리스트는 False로 평가
  items = []
  if items:  # bool(items)와 동일
      print("항목이 있습니다")
  else:
      print("항목이 없습니다")  # 이게 출력됨
  ```

* __'bool()' 함수를 이용해서 '객체'를 '불리언'으로 변환할 수 있다.__

  예를 들어 'bool(10)'은 'True'를 반환한다. 거꾸로 'bool(0)'은 'False'를 반환한다.
  '0'이 아닌 모든 숫자는 'True'로 간주된다. '빈 문자열'은 'False'로 간주되고 '비어 있지 않은 문자열'은 'True'로 간주된다.
  '빈 리스트'나 '빈 딕셔너리'도 'False'로 간주된다. '빈 튜플'도 'False'로 간주된다.
  'None'은 'False'로 간주된다. 'None'이 아닌 일반 객체 참조 값은 'True'로 간주된다.

* __'int()' 함수를 이용해서 '객체'를 '정수'로 변환할 수 있다.__

  예를 들어 'int(10.5)'은 '10'을 반환한다.

* __'float()' 함수를 이용해서 '객체'를 '실수'로 변환할 수 있다.__

  예를 들어 'float(10)'은 '10.0'을 반환한다.

* __'list()' 함수를 이용해서 '객체'를 '리스트'로 변환할 수 있다.__

  예를 들어 'list("hello")'는 '['h', 'e', 'l', 'l', 'o']'를 반환한다.

* __'range()' 함수를 이용해서 '범위'를 생성할 수 있다.__

  range() 함수는 세 가지 사용법이 있다:
    1. range(stop): 0부터 시작해서 stop-1까지의 정수 시퀀스를 생성한다.
    2. range(start, stop): start부터 시작해서 stop-1까지의 정수 시퀀스를 생성한다.
    3. range(start, stop, step): start부터 시작해서 stop-1까지 step만큼 증가하는 정수 시퀀스를 생성한다.
       * step은 음수도 가능하다. 이 경우 감소하는 시퀀스가 생성된다.

  range() 함수의 특징:

  * 메모리 효율적: 실제로 모든 숫자를 저장하지 않고, 필요할 때 생성한다.
  * 불변(immutable) 시퀀스 타입이다.
  * for 루프에서 자주 사용된다.

  예제 코드:
  
    ```python
    # 기본 사용법 (0부터 4까지)
    for i in range(5):
        print(i)    # 출력: 0, 1, 2, 3, 4

    # 시작과 끝 지정 (1부터 4까지)
    for i in range(1, 5):
        print(i)    # 출력: 1, 2, 3, 4

    # 증가값 지정 (1부터 9까지 2씩 증가)
    for i in range(1, 10, 2):
        print(i)    # 출력: 1, 3, 5, 7, 9

    # 감소하는 시퀀스 (-2씩 감소)
    for i in range(10, 0, -2):
        print(i)    # 출력: 10, 8, 6, 4, 2
    ```

    파이썬에서는 다음과 같은 C/C++의 전통적인 for 문을 사용할 수 없다:

    ```c
    // C/C++ 스타일의 전통적인 for 문
    for (int i = 0; i < 5; ++i) {
        printf("%d\n", i);    // 출력: 0, 1, 2, 3, 4
    }
    ```

    파이썬에서는 대신에 앞선 예제에서와 같이 range() 함수를 활용한다.
