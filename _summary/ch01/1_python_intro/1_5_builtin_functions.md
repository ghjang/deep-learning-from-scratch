# 1.5 내장 전역 함수

> [목차로 돌아가기](../../README.md) | [이전: None과 문법적 특이사항](./1_5_none_and_syntax.md)

## 1.5.1 타입 관련 함수

* __'type()' 함수를 이용하면 주어진 '객체'의 '타입'을 알 수 있다.__

  예를 들어 'type(10)'은 'int'를 반환한다.
  
  ```python
  print(type(42))        # <class 'int'>
  print(type("hello"))   # <class 'str'>
  print(type([1, 2, 3])) # <class 'list'>
  ```

* __'isinstance()' 함수를 이용해서 주어진 '객체'가 '특정 타입'인지 확인할 수 있다.__

  예를 들어 'isinstance(10, int)'는 'True'를 반환한다.
  
  ```python
  print(isinstance(42, int))         # True
  print(isinstance("hello", str))    # True
  print(isinstance("hello", int))    # False
  
  # 여러 타입 중 하나인지 확인
  print(isinstance(42, (int, float)))  # True
  
  # 상속 관계도 확인
  class Parent:
      pass
      
  class Child(Parent):
      pass
      
  obj = Child()
  print(isinstance(obj, Child))   # True
  print(isinstance(obj, Parent))  # True
  ```

* __'id()' 함수를 이용해서 '객체'의 '메모리 주소'를 알 수 있다.__

  예를 들어 'id(10)'은 '객체 10'의 '메모리 주소'를 반환한다.
  
  ```python
  x = [1, 2, 3]
  y = x
  z = [1, 2, 3]
  
  print(id(x))  # 예: 140233683913024
  print(id(y))  # 예: 140233683913024 (x와 동일)
  print(id(z))  # 예: 140233683913344 (다른 객체)
  ```

## 1.5.2 객체 검사 및 변환 함수

* __'dir()' 함수를 이용해서 '객체'의 '속성'과 '메서드'를 알 수 있다.__

  예를 들어 'dir(10)'은 'int' 타입의 '객체'가 가지고 있는 '속성'과 '메서드'를 반환한다.
  
  ```python
  print(dir(42))        # int의 모든 메서드와 속성
  print(dir("hello"))   # str의 모든 메서드와 속성
  
  # 사용자 정의 클래스에도 적용 가능
  class MyClass:
      def __init__(self):
          self.x = 10
          self.y = 20
      
      def my_method(self):
          pass
  
  obj = MyClass()
  print(dir(obj))  # 모든 속성과 메서드 목록 출력
  ```

* __'len()' 함수를 이용해서 '리스트'나 '문자열'의 길이를 알 수 있다.__

  예를 들어 'len([1, 2, 3])'은 '3'을 반환한다.
  
  ```python
  print(len([1, 2, 3]))     # 3
  print(len("hello"))       # 5
  print(len({"a": 1, "b": 2}))  # 2
  
  # 사용자 정의 객체도 __len__ 메서드를 구현하면 사용 가능
  class MyClass:
      def __len__(self):
          return 42
  
  obj = MyClass()
  print(len(obj))  # 42
  ```

* __'str()' 함수를 이용해서 '객체'를 '문자열'로 변환할 수 있다.__

  예를 들어 'str(10)'은 '10'을 반환한다.
  
  ```python
  print(str(42))      # "42"
  print(str([1, 2]))  # "[1, 2]"
  ```

* __'print()' 함수를 이용해서 '객체'를 표준 출력으로 출력할 수 있다.__

  예를 들어 'print(10)'은 '숫자 10'을 '문자열 10'으로 변환해서 표준 출력 장치로 출력한다.
  
  ```python
  print(42)           # 42
  print("hello")      # hello
  print(1, 2, 3)      # 1 2 3
  print(1, 2, sep=", ")  # 1, 2
  print("hello", end="!") # hello! (줄바꿈 없음)
  ```

* __'repr()' 함수를 이용해서 '객체'를 '개발자를 위한 문자열'로 변환할 수 있다.__

  예를 들어 'repr(10)'은 '10'을 반환한다. 'str()' 함수와 다르게 'repr()' 함수는 '개발자를 위한 문자열'을 반환한다.
  
  ```python
  s = "Hello\nWorld"
  print(str(s))   # Hello
                  # World     (줄바꿈 실행됨)
  print(repr(s))  # 'Hello\nWorld'  (이스케이프 문자가 보임)
  ```

## 1.5.3 문자열 출력과 표현

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

## 1.5.4 불리언 변환과 형변환

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

* __'int()' 함수를 이용해서 '객체'를 '정수'로 변환할 수 있다.__

  예를 들어 'int(10.5)'은 '10'을 반환한다.
  
  ```python
  print(int(10.9))    # 10 (소수점 버림)
  print(int("42"))    # 42
  print(int("0x2A", 16))  # 42 (16진수 변환)
  print(int("0b101010", 2))  # 42 (2진수 변환)
  ```

* __'float()' 함수를 이용해서 '객체'를 '실수'로 변환할 수 있다.__

  예를 들어 'float(10)'은 '10.0'을 반환한다.
  
  ```python
  print(float(42))    # 42.0
  print(float("42"))  # 42.0
  print(float("42.5"))  # 42.5
  print(float("1e-3"))  # 0.001
  ```

* __'list()' 함수를 이용해서 '객체'를 '리스트'로 변환할 수 있다.__

  예를 들어 'list("hello")'는 '['h', 'e', 'l', 'l', 'o']'를 반환한다.
  
  ```python
  print(list("hello"))      # ['h', 'e', 'l', 'l', 'o']
  print(list((1, 2, 3)))    # [1, 2, 3]
  print(list(range(3)))     # [0, 1, 2]
  print(list({"a": 1, "b": 2}))  # ['a', 'b'] (키만 리스트로 변환)
  ```

## 1.5.5 범위 생성 함수

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

> [목차로 돌아가기](../../README.md) | [이전: None과 문법적 특이사항](./1_5_none_and_syntax.md)
