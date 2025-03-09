# 1.5 None과 문법적 특이사항

> [목차로 돌아가기](../README.md) | [이전: 객체 지향 특성과 특수 연산자](./1_4_oop_and_operators.md) | [다음: 파이썬 내장 전역 함수](./1_6_builtin_functions.md)

## 1.5.1 None과 NoneType

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

## 1.5.2 문법적 특이사항

* __'...'은 'pass'를 의미한다.__

  'pass'는 '아무것도 하지 않는 문장'이다. 예를 들어서 'if' 문에서 '아무것도 하지 않을 때' 사용할 수 있다:

    ```python
    if x < 0:
        print('negative')
    else:
        ... # 올바른 파이썬 문법이다.
    ```

  pass와 마찬가지로 Ellipsis(...)는 문법적으로 유효한 표현식이지만 아무 작업도 수행하지 않습니다.
  
  ```python
  def todo_function():
      ...  # 아직 구현하지 않은 함수
  
  class TodoClass:
      ...  # 아직 구현하지 않은 클래스
  ```

* __타입 힌트에서의 '...'(Ellipsis)__

  파이썬의 타입 힌트에서 '...'는 재귀적 타입이나 지연 평가 타입을 표현하는데 사용됩니다:
  
  ```python
  # 재귀적 타입 정의
  from typing import List, Dict, Union
  
  # 중첩된 JSON과 같은 구조 표현
  JSONValue = Union[str, int, float, bool, None, List['JSONValue'], Dict[str, 'JSONValue']]
  
  # 또는 3.10 이후의 타입 별칭 사용
  type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
  ```

> [목차로 돌아가기](../README.md) | [이전: 객체 지향 특성과 특수 연산자](./1_4_oop_and_operators.md) | [다음: 파이썬 내장 전역 함수](./1_6_builtin_functions.md)
