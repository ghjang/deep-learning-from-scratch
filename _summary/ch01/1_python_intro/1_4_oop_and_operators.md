# 1.4 객체 지향 특성과 특수 연산자

> [목차로 돌아가기](./README.md) | [이전: 가변성과 시퀀스](./1_3_mutability_and_sequences.md) | [다음: None과 문법적 특이사항](./1_5_none_and_syntax.md)

## 1.4.1 객체 지향 특성

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

## 1.4.2 특수 연산자

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

> [목차로 돌아가기](./README.md) | [이전: 가변성과 시퀀스](./1_3_mutability_and_sequences.md) | [다음: None과 문법적 특이사항](./1_5_none_and_syntax.md)
