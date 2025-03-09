# 1. 파인썬 인트로

'1장 헬로 파이썬'을 읽고 기억해야할만 한 내용들을 정리한다.

## 1.1 주의사항

__파이썬 '3' 버전은 '2' 버전과 호환되지 않는다.__ 즉 '하위 호환성'이 없다. 기본적인 언어 문법은 바뀌지 않았지만, 언어가 발전하면서 새로운 기능이 추가되거나 기존 기능이 변경되기도 했다.

* __'/' 연산자를 이용한 나눗셈의 결과가 '타입'이 달라졌다.__

  '2' 버전에서는 '정수 나누기 정수'의 결과가 '정수'로 나오지만, '3' 버전에서는 '실수'로 나온다. 예를 들어서 '2' 버전에서 '7 / 5'는 '1'이지만, '3' 버전에서는 '1.4'이다.

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

* __'**' 연산자는 '거듭제곱'을 나타낸다.__

  예를 들어 '2 ** 3'은 '2의 3제곱'을 의미한다.

* __'//' 연산자는 '나눗셈의 몫'을 나타낸다.__

  예를 들어 '7 // 5'는 '7을 5로 나눈 몫'을 의미한다.

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
