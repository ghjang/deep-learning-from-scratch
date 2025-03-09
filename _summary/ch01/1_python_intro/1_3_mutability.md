# 1.3 가변(Mutable)과 불변(Immutable) 타입

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 기본 자료형](./1_2_data_types.md) | [다음: 시퀀스 타입과 슬라이싱](./1_4_sequence_types.md)

## 1.3.1 가변(Mutable)과 불변(Immutable) 타입

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

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 기본 자료형](./1_2_data_types.md) | [다음: 시퀀스 타입과 슬라이싱](./1_4_sequence_types.md)
