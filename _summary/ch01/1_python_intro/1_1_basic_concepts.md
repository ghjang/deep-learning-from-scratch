# 1.1 파이썬 기본

> [목차로 돌아가기](../../README.md) | [다음: 파이썬 기본 자료형](./1_2_data_types.md)

## 1.1.1 파이썬 2와 3의 주요 차이점

__파이썬 '3' 버전은 '2' 버전과 호환되지 않는다.__ 즉 '하위 호환성'이 없다. 기본적인 언어 문법은 바뀌지 않았지만, 언어가 발전하면서 새로운 기능이 추가되거나 기존 기능이 변경되기도 했다.

* __'/' 연산자를 이용한 나눗셈의 결과가 '타입'이 달라졌다.__

  '2' 버전에서는 '정수 나누기 정수'의 결과가 '정수'로 나오지만, '3' 버전에서는 '실수'로 나온다. 예를 들어서 '2' 버전에서 '7 / 5'는 '1'이지만, '3' 버전에서는 '1.4'이다. 또한 '3' 버전에서 '정수 / 정수'의 결과가 '정수'이어도 결과 타입이 '실수'로 바뀌었다.

* __'int' 기본 타입의 숫자 표현 범위가 무한대로 바뀌었다.__

  '2' 버전에서는 '정수'를 표현하는 자료 타입으로 'int'와 'long'이 있었다. 'int'는 '32비트'나 '64비트'로 제한되어 있었고, 'long'은 '무한대'로 표현할 수 있었다. '3' 버전에서는 'int' 자료형만 있고, '메모리가 허용하는 한 무제한으로 큰 정수'를 표현할 수 있다.

* __'print'가 '문'에서 '함수'로 변경되었다.__

  파이썬 2에서는 print가 문법적 키워드로서 괄호 없이 사용되었지만, 파이썬 3에서는 일반 함수로 변경되어 반드시 괄호가 필요하다:
  
  ```python
  # 파이썬 2
  print "Hello, World!"
  
  # 파이썬 3
  print("Hello, World!")
  ```

* __문자열과 바이트 타입이 엄격하게 분리되었다.__

  파이썬 2에서는 문자열(str)이 바이트 시퀀스였지만, 파이썬 3에서는 문자열은 유니코드 문자의 시퀀스이고 바이트는 별도의 bytes 타입으로 분리되었다:
  
  ```python
  # 파이썬 3
  s = "안녕하세요"  # 유니코드 문자열
  b = b"hello"     # 바이트 시퀀스 (ASCII만 가능)
  
  # 변환이 필요함
  encoded = s.encode('utf-8')  # 문자열 → 바이트
  decoded = b.decode('utf-8')  # 바이트 → 문자열
  ```

* __'xrange'가 삭제되고 'range'가 제너레이터처럼 동작하도록 변경되었다.__

  파이썬 2의 range는 목록을 반환했지만, 파이썬 3의 range는 메모리 효율적인 이터레이터처럼 동작한다(파이썬 2의 xrange와 유사). 큰 범위를 생성할 때 메모리를 절약할 수 있다:
  
  ```python
  # 파이썬 2
  numbers = range(10000000)  # 실제 리스트 생성 (메모리에 저장)
  
  # 파이썬 3
  numbers = range(10000000)  # 범위 객체만 생성 (값은 필요할 때 생성)
  ```

* __'Exception' 처리 문법이 변경되었다.__

  예외를 처리할 때 'as' 키워드 사용이 필수적으로 바뀌었고, 예외 객체의 관리 방식도 변경되었다:
  
  ```python
  # 파이썬 2
  try:
      do_something()
  except ValueError, e:  # 쉼표로 예외 객체 지정
      print e
  
  # 파이썬 3
  try:
      do_something()
  except ValueError as e:  # as 키워드 필수
      print(e)
  ```

## 1.1.2 특수 연산자와 구문 요소

* __'**' 연산자는 '거듭제곱'을 나타낸다.__

  예를 들어 '2 ** 3'은 '2의 3제곱'을 의미한다.

* __'//' 연산자는 '나눗셈의 몫'을 나타낸다.__

  예를 들어 '7 // 5'는 '7을 5로 나눈 몫'을 의미한다.

* __':='(왈러스 연산자)는 할당과 표현식을 동시에 수행한다.__

  파이썬 3.8부터 도입된 왈러스 연산자는 변수에 값을 할당하면서 동시에 그 표현식을 평가할 수 있게 해준다:

  ```python
  # 일반적인 방식
  data = get_data()
  if data:
      process(data)
      
  # 왈러스 연산자 사용
  if data := get_data():
      process(data)
      
  # 반복문에서의 활용
  while chunk := file.read(8192):
      process(chunk)
      
  # 리스트 컴프리헨션에서 활용
  results = [transformed for x in data if (transformed := transform(x)) is not None]
  ```

* __'is'는 객체 식별자 비교를 수행하는 키워드이다.__

  'is'는 두 변수가 메모리상에서 동일한 객체를 참조하는지 비교하는 키워드이다.
  
  * '==' 연산자: 논리적 동등성(logical equality) 비교 - 두 객체의 값이 같은지 비교
  * 'is' 키워드: 물리적 동일성(physical identity) 비교 - 두 변수가 실제로 메모리 상 같은 객체를 가리키는지 비교
  
  ```python
  a = [1, 2, 3]
  b = [1, 2, 3]
  c = a
  
  print(a == b)  # True (값이 같음)
  print(a is b)  # False (다른 객체)
  print(a is c)  # True (같은 객체)
  ```

* __'yield'는 제너레이터 함수를 만드는 키워드이다.__

  `yield`는 함수가 값을 반환하면서도 실행 상태를 유지하게 해주는 키워드로, 제너레이터 함수를 정의하는 데 사용된다:

  ```python
  # 기본적인 제너레이터 함수
  def count_up_to(max):
      count = 1
      while count <= max:
          yield count  # 값을 반환하고 함수 상태 저장
          count += 1   # 다음 호출 시 여기서부터 재개
          
  # 제너레이터 사용
  counter = count_up_to(5)  # 제너레이터 객체 생성 (아직 실행되지 않음)
  
  print(next(counter))  # 1 (첫 번째 yield까지 실행)
  print(next(counter))  # 2 (두 번째 yield까지 실행)
  
  # for 루프로 나머지 값 소비
  for number in counter:
      print(number)     # 3, 4, 5 출력
  ```

  제너레이터는 다음과 같은 특징을 갖는다:
  
  1. __메모리 효율성__: 모든 결과를 한 번에 메모리에 저장하지 않고 필요할 때만 계산
  2. __지연 평가__: 요청할 때만 다음 값을 생성 (lazy evaluation)
  3. __상태 유지__: 함수의 실행 상태가 유지되어 다음 호출 시 이어서 실행
  
  ```python
  # 대용량 데이터 처리 예시
  def read_large_file(file_path, chunk_size=1024):
      with open(file_path, 'r') as file:
          while True:
              data = file.read(chunk_size)
              if not data:  # 파일의 끝에 도달
                  break
              yield data
                  
  # 수 기가바이트 파일도 적은 메모리로 처리 가능
  for chunk in read_large_file('huge_log.txt'):
      process_data(chunk)
  ```
  
  `yield from` 구문으로 다른 이터러블의 값을 위임해서 yield할 수 있다:
  
  ```python
  def chain(*iterables):
      for it in iterables:
          yield from it  # it의 각 항목을 하나씩 yield
          
  # 사용 예시
  result = list(chain([1, 2], [3, 4, 5], [6]))
  print(result)  # [1, 2, 3, 4, 5, 6]
  ```

* __언패킹 연산자 '*', '**'는 시퀀스와 매핑을 풀어준다.__

  파이썬에서는 두 가지 유형의 언패킹 연산자가 있다:

  1. `*` 연산자 - 시퀀스 언패킹(리스트, 튜플, 문자열 등):
  
     ```python
     # 시퀀스 분해하여 변수에 할당
     first, *middle, last = [1, 2, 3, 4, 5]
     print(first)   # 1
     print(middle)  # [2, 3, 4]
     print(last)    # 5
     
     # 함수 호출 시 시퀀스를 개별 인자로 확장
     def add(a, b, c):
         return a + b + c
         
     values = [1, 2, 3]
     print(add(*values))  # 6  (add(1, 2, 3)와 동일)
     
     # 리스트 병합
     list1 = [1, 2, 3]
     list2 = [4, 5]
     combined = [*list1, *list2]
     print(combined)  # [1, 2, 3, 4, 5]
     ```
  
  2. `**` 연산자 - 딕셔너리 언패킹(키-값 쌍):
  
     ```python
     # 딕셔너리를 키워드 인자로 확장
     def create_profile(name, age, job):
         return f"{name}, {age}, {job}"
     
     data = {"name": "Kim", "age": 30, "job": "Developer"}
     profile = create_profile(**data)  # create_profile(name="Kim", age=30, job="Developer")와 동일
     
     # 딕셔너리 병합 (Python 3.5+)
     defaults = {"color": "red", "size": "medium"}
     options = {"size": "large", "material": "cotton"}
     settings = {**defaults, **options}
     print(settings)  # {'color': 'red', 'size': 'large', 'material': 'cotton'}
     # 중복 키는 나중에 오는 딕셔너리의 값으로 덮어씌워짐
     ```

  3. 함수 정의에서의 언패킹 - 가변 인자 처리:
  
     ```python
     # *args: 가변 길이 위치 인자
     def sum_all(*args):
         """임의 개수의 인자를 받아 모두 더하는 함수"""
         total = 0
         for num in args:
             total += num
         return total
     
     print(sum_all(1, 2, 3))        # 6
     print(sum_all(10, 20, 30, 40)) # 100
     
     # **kwargs: 가변 길이 키워드 인자
     def print_info(**kwargs):
         """임의 개수의 키워드 인자를 받아 출력하는 함수"""
         for key, value in kwargs.items():
             print(f"{key}: {value}")
     
     print_info(name="Kim", age=30, job="Developer")
     # 출력:
     # name: Kim
     # age: 30
     # job: Developer
     
     # *args와 **kwargs 함께 사용
     def flexible_function(*args, **kwargs):
         """위치 인자와 키워드 인자를 모두 받는 유연한 함수"""
         print(f"위치 인자: {args}")
         print(f"키워드 인자: {kwargs}")
     
     flexible_function(1, 2, 3, name="Test", value=True)
     # 출력:
     # 위치 인자: (1, 2, 3)
     # 키워드 인자: {'name': 'Test', 'value': True}
     ```

     __일반 파라미터와 언패킹 파라미터 혼합:__

     파이썬에서 함수 정의 시 일반 파라미터와 언패킹 파라미터를 함께 사용할 때는 엄격한 순서 규칙을 따라야 한다:

     ```python
     # 파라미터 순서 규칙
     def complex_function(
         pos1, pos2,              # 1. 일반 위치 파라미터
         /,                       # 위치 전용 파라미터 구분자 (3.8+)
         pos_or_kwd1, pos_or_kwd2,# 2. 위치 또는 키워드로 전달 가능한 파라미터
         *args,                   # 3. 추가 위치 인자 (튜플로 수집)
         kwd1, kwd2,              # 4. 키워드 전용 파라미터 (키워드로만 전달 가능)
         **kwargs                 # 5. 추가 키워드 인자 (딕셔너리로 수집)
     ):
         print(f"위치 전용: {pos1}, {pos2}")
         print(f"위치 또는 키워드: {pos_or_kwd1}, {pos_or_kwd2}")
         print(f"추가 위치 인자: {args}")
         print(f"키워드 전용: {kwd1, kwd2}")
         print(f"추가 키워드 인자: {kwargs}")
     
     # 호출 예시
     complex_function(
         1, 2,                    # pos1, pos2 (위치 전용)
         "a", "b",                # pos_or_kwd1, pos_or_kwd2
         "extra1", "extra2",      # *args로 수집됨
         kwd1="k1", kwd2="k2",    # kwd1, kwd2 (키워드 전용)
         extra_kwd="extra_val"    # **kwargs로 수집됨
     )
     ```

     __주의사항:__

     1. __순서 규칙 위반은 SyntaxError 발생__:

        ```python
        # 잘못된 예: *args가 키워드 전용 인자보다 뒤에 위치
        def wrong_order(a, b, kwd_only=None, *args):  # 구문 오류 발생!
            pass
        ```

     2. __기본값은 위치 전용이나 위치/키워드 인자에만 지정 가능__:

        ```python
        # 올바른 예
        def func(a, b=10, *, c, d=20):
            pass
            
        func(1, c=3)  # a=1, b=10, c=3, d=20
        ```

     3. __파이썬 3.8부터 위치 전용 파라미터 도입__:

        `/` 구분자 이전의 파라미터는 반드시 위치 인자로만 전달해야 함

        ```python
        def pos_only_func(x, y, /, z):
            print(x, y, z)
            
        pos_only_func(1, 2, z=3)    # 가능
        pos_only_func(1, 2, 3)      # 가능
        pos_only_func(x=1, y=2, z=3)# 오류! x, y는 위치 전용
        ```

  이 언패킹 연산자들은 코드를 더 간결하게 만들고, 가변 길이의 인자를 다룰 때 유용하다.

* __'as'는 컨텍스트에 따라 다른 역할을 하는 키워드이다.__

  'as'는 파이썬에서 컨텍스트에 따라 다른 의미로 사용되는 키워드이다. 같은 키워드지만 사용되는 위치와 상황에 따라 다른 기능을 수행한다:
  
  1. import 문에서 - 모듈이나 객체에 별칭(alias)을 부여:

     ```python
     import numpy as np
     from datetime import datetime as dt
     ```
  
  2. with 문에서 - 컨텍스트 관리자의 반환값을 변수에 바인딩:

     ```python
     with open('file.txt') as file:
         content = file.read()
     ```
  
  3. except 절에서 - 발생한 예외 객체를 변수에 바인딩:

     ```python
     try:
         result = 1 / 0
     except ZeroDivisionError as error:
         print(f"에러 발생: {error}")
     ```
  
  4. 패턴 매칭(Python 3.10+)에서 - 매칭된 값에 이름 부여:

     ```python
     match command:
         case ["quit" | "exit" as cmd]:
             print(f"{cmd} 명령으로 종료합니다")
         case ["save" as action, filename]:
             print(f"{action}: {filename}에 저장합니다")
     ```

  이러한 방식으로 'as' 키워드는 다양한 문맥에서 "이름 바인딩"이라는 일관된 개념을 유지하면서도 상황에 맞는 특화된 역할을 수행한다.
  
* __컨텍스트 매니저(with 문)는 자원 관리를 자동화한다.__

  컨텍스트 매니저는 파이썬의 "스코프 가드" 역할을 하는 구문으로, 리소스 획득과 해제를 자동화한다. 내부적으로는 `__enter__`와 `__exit__` 메서드 쌍으로 동작한다:

  ```python
  # 파일 자동 닫기
  with open('file.txt', 'r') as file:  # __enter__ 호출: 파일 열기, 반환값을 file에 바인딩
      data = file.read()
  # 블록을 벗어날 때 자동으로 __exit__ 호출: 파일 닫기
  ```

  컨텍스트 매니저의 내부 동작 흐름:
  1. `with` 문 시작: 표현식의 `__enter__` 메서드 호출
  2. `__enter__` 메서드의 반환 값을 `as` 뒤의 변수에 할당
  3. 블록 내 코드 실행
  4. 블록 종료 시 __어떤 상황에서든__ `__exit__` 메서드 호출
     * 정상 종료: `__exit__(None, None, None)` 호출
     * 예외 발생: `__exit__(exc_type, exc_value, traceback)` 호출

  ```python
  # 예외 처리 자동화 예시
  class Transaction:
      def __enter__(self):
          print("트랜잭션 시작")
          return self
          
      def __exit__(self, exc_type, exc_val, exc_tb):
          if exc_type is None:
              print("트랜잭션 커밋")
              return True  # 정상 종료
          else:
              print(f"오류 발생: {exc_type}, 롤백 수행")
              return False  # 예외 전파 (True 반환 시 예외 억제)
  
  # 정상 케이스
  with Transaction() as tx:
      print("작업 수행")
  # 출력:
  # 트랜잭션 시작
  # 작업 수행
  # 트랜잭션 커밋
  
  # 오류 케이스
  try:
      with Transaction() as tx:
          print("작업 시작")
          raise ValueError("오류 발생!")  # 의도적 예외
  except ValueError:
      print("예외 처리됨")
  # 출력:
  # 트랜잭션 시작
  # 작업 시작
  # 오류 발생: <class 'ValueError'>, 롤백 수행
  # 예외 처리됨
  ```
  
  contextmanager 데코레이터를 사용하여 컨텍스트 매니저를 더 간단하게 구현해 사용할 수 있는 경우도 있다:
  
  ```python
  from contextlib import contextmanager
  
  @contextmanager
  def managed_resource():
      print("리소스 획득")    # __enter__ 부분
      try:
          yield "리소스"      # as 뒤 변수에 할당되는 값
      finally:                # 항상 실행됨 (__exit__ 부분)
          print("리소스 해제") # 예외가 발생해도 반드시 실행
          
  with managed_resource() as resource:
      print(f"리소스 사용 중: {resource}")
  ```
  
  컨텍스트 매니저는 파일 핸들링, 데이터베이스 연결, 락(lock) 관리, 트랜잭션 제어 등 "획득-사용-해제" 패턴이 필요한 상황에서 코드 안전성을 높여준다.

* __'...'(Ellipsis)는 'pass'와 유사하게 사용된다.__

  'pass'는 '아무것도 하지 않는 문장'이다. 예를 들어서 'if' 문에서 '아무것도 하지 않을 때' 사용할 수 있다:

    ```python
    if x < 0:
        print('negative')
    else:
        ... # 올바른 파이썬 문법이다.
    ```

  pass와 마찬가지로 Ellipsis(...)는 문법적으로 유효한 표현식이지만 아무 작업도 수행하지 않는다.
  
  ```python
  def todo_function():
      ...  # 아직 구현하지 않은 함수
  
  class TodoClass:
      ...  # 아직 구현하지 않은 클래스
  ```

* __'match-case'는 파이썬의 패턴 매칭 구문이다.__

  파이썬 3.10부터 도입된 `match-case`는 다른 언어의 `switch-case`와 유사하지만 더 강력한 구조적 패턴 매칭 기능을 제공한다:

  ```python
  # 기본 사용법
  def process_command(command):
      match command:
          case "quit":
              return "종료합니다"
          case "help":
              return "도움말을 표시합니다"
          case _:  # 와일드카드 패턴 (default 역할)
              return f"알 수 없는 명령: {command}"
  
  # 시퀀스 패턴
  def process_point(point):
      match point:
          case (0, 0):
              return "원점"
          case (0, y):
              return f"y축 위의 점 (0, {y})"
          case (x, 0):
              return f"x축 위의 점 ({x}, 0)"
          case (x, y):
              return f"좌표 ({x}, {y})"
          case _:
              return "유효한 좌표가 아님"
  
  # 클래스 패턴
  class Point:
      def __init__(self, x, y):
          self.x = x
          self.y = y
  
  def locate_point(point):
      match point:
          case Point(x=0, y=0):
              return "원점"
          case Point(x=0, y=y):
              return f"y축 위의 점"
          case Point(x=x, y=0):
              return f"x축 위의 점"
          case Point():
              return "일반 좌표"
          case _:
              return "점이 아님"
  
  # 가드 조건 사용
  def check_value(value):
      match value:
          case int(n) if n < 0:
              return "음수 정수"
          case int(n) if n > 0:
              return "양수 정수"
          case int(0):
              return "0"
          case float(f):
              return "실수"
          case str(s):
              return f"문자열: {s}"
          case _:
              return "기타 타입"
  ```

  `match-case`는 단순 값 비교뿐만 아니라 구조 분해, 타입 검사, 조건부 매칭 등을 모두 지원하여 복잡한 데이터 구조를 효과적으로 처리할 수 있다.

## 1.1.3 타입 힌트와 타입 시스템

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

* __재귀적 타입 정의가 가능하다.__

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

* __타입 힌트에서 '...'(Ellipsis) 활용하기__

  파이썬의 타입 힌트에서 '...'는 다음과 같은 용도로 사용된다:
  
  1. 가변 길이 튜플 타입 정의:

      ```python
      from typing import Tuple
      
      # ... 사용으로 임의 개수의 int 요소를 가진 튜플 표현
      Vector = Tuple[int, ...]
      
      def process_vector(v: Vector) -> int:
          return sum(v)
          
      # 사용 예시
      process_vector((1, 2))          # 유효
      process_vector((1, 2, 3, 4, 5)) # 유효
      ```
  
  1. 타입 체킹에서 "구현 예정" 표시:

      ```python
      def calculate_something(x: int, y: int) -> ...:
          # 반환 타입이 아직 결정되지 않았거나 복잡한 경우 
          # ... 사용으로 "추후 정의될 예정" 표시
          return x + y
      ```

> [목차로 돌아가기](../../README.md) | [다음: 파이썬 기본 자료형](./1_2_data_types.md)
