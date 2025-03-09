# 1.4 객체 지향 특성

> [목차로 돌아가기](../../README.md) | [이전: 시퀀스 타입과 슬라이싱](./1_4_sequence_types.md) | [다음: None과 문법적 특이사항](./1_6_none_and_syntax.md)

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

> [목차로 돌아가기](../../README.md) | [이전: 시퀀스 타입과 슬라이싱](./1_4_sequence_types.md) | [다음: None과 문법적 특이사항](./1_6_none_and_syntax.md)
