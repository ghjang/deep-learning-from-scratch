# 1.5 None 객체

> [목차로 돌아가기](../../README.md) | [이전: 객체 지향 특성과 특수 연산자](./1_4_oop_and_operators.md) | [다음: 파이썬 내장 전역 함수](./1_6_builtin_functions.md)

## 1.5.1 None 객체의 특성

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

## 1.5.2 None의 활용

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

> [목차로 돌아가기](../../README.md) | [이전: 객체 지향 특성과 특수 연산자](./1_4_oop_and_operators.md) | [다음: 파이썬 내장 전역 함수](./1_6_builtin_functions.md)
