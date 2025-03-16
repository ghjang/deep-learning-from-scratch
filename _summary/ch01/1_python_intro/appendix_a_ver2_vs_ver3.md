# 부록 A: 파이썬 2와 3의 주요 차이점

> [목차로 돌아가기](../../README.md) | [이전: 시퀀스 타입과 슬라이싱](./1_9_sequence_types.md) | [다음: 파이썬 Pickle 객체 직렬화](./appendix_b_pickle.md)

__파이썬 '3' 버전은 '2' 버전과 호환되지 않는다.__ 즉 '하위 호환성'이 없다. 기본적인 언어 문법은 바뀌지 않았지만, 언어가 발전하면서 새로운 기능이 추가되거나 기존 기능이 변경되기도 했다.

## 주요 차이점 요약

### 나눗셈 연산자 [A.1](#a1-나눗셈-연산자-변경)

- 파이썬 2: `7 / 5` → `1` (정수)
- 파이썬 3: `7 / 5` → `1.4` (실수)

### 정수 타입 [A.2](#a2-정수-타입-통합)

- 파이썬 2: `int`(제한적), `long`(무제한)
- 파이썬 3: 단일 `int` 타입(무제한)

### print 구문 [A.3](#a3-print-함수화)

- 파이썬 2: `print "Hello"` (문장)
- 파이썬 3: `print("Hello")` (함수)

### 문자열 [A.4](#a4-문자열과-바이트-타입-분리)

- 파이썬 2: 기본 str은 바이트 시퀀스
- 파이썬 3: 기본 str은 유니코드, 바이트는 별도 타입

### 유니코드 [A.5](#a5-유니코드-처리-개선)

- 파이썬 2: `u"문자열"` 형식 필요
- 파이썬 3: 모든 문자열이 유니코드

### 범위 생성 [A.6](#a6-range-동작-방식-변경)

- 파이썬 2: `range()`, `xrange()`
- 파이썬 3: `range()`만 존재(제너레이터 방식)

### 예외 처리 [A.7](#a7-예외-처리-문법-변경)

- 파이썬 2: `except Error, e:`
- 파이썬 3: `except Error as e:`

### 사용자 입력 [A.8](#a8-입력-함수-통합)

- 파이썬 2: `input()`과 `raw_input()`
- 파이썬 3: `input()`만 존재

### 딕셔너리 뷰 [A.9](#a9-딕셔너리-뷰-객체-도입)

- 파이썬 2: `.keys()` 등이 리스트 반환
- 파이썬 3: `.keys()` 등이 뷰 객체 반환

### 타입간 정렬 [A.10](#a10-정렬-동작-변경)

- 파이썬 2: 다른 타입 정렬 가능
- 파이썬 3: 타입 비교 불가능 시 에러

### 내장 함수 반환 [A.11](#a11-이터레이터-반환-기본화)

- 파이썬 2: 대부분 리스트 반환
- 파이썬 3: 대부분 이터레이터 반환

## A.1 나눗셈 연산자 변경

파이썬 2와 3에서는 `/` 연산자를 이용한 나눗셈의 결과가 다르게 처리된다.

```python
# 파이썬 2
print(7 / 5)  # 1
print(7 / 5.0)  # 1.4

# 파이썬 3
print(7 / 5)  # 1.4
print(7 // 5)  # 1 (정수 나눗셈은 // 연산자 사용)
```

파이썬 2에서는 '정수 나누기 정수'의 결과가 '정수'로 나오지만, 파이썬 3에서는 '실수'로 나온다. 예를 들어서 파이썬 2에서 '7 / 5'는 '1'이지만, 파이썬 3에서는 '1.4'이다. 또한 파이썬 3에서 '정수 / 정수'의 결과가 '정수'이어도 결과 타입이 '실수'로 바뀌었다.

## A.2 정수 타입 통합

파이썬 2와 3에서 정수 타입의 처리 방식이 변경되었다.

```python
# 파이썬 2
big_num = 12345678901234567890
print(type(big_num))  # <type 'long'>

# 파이썬 3
big_num = 12345678901234567890
print(type(big_num))  # <class 'int'>
```

파이썬 2에서는 '정수'를 표현하는 자료 타입으로 'int'와 'long'이 있었다. 'int'는 '32비트'나 '64비트'로 제한되어 있었고, 'long'은 '무한대'로 표현할 수 있었다. 파이썬 3에서는 'int' 자료형만 있고, '메모리가 허용하는 한 무제한으로 큰 정수'를 표현할 수 있다.

## A.3 print 함수화

파이썬 2와 3에서 `print` 구문이 다르게 처리된다.

```python
# 파이썬 2
print "Hello, World!"

# 파이썬 3
print("Hello, World!")
```

파이썬 2에서는 print가 문법적 키워드로서 괄호 없이 사용되었지만, 파이썬 3에서는 일반 함수로 변경되어 반드시 괄호가 필요하다.

## A.4 문자열과 바이트 타입 분리

파이썬 2와 3에서는 문자열과 바이너리 데이터의 처리가 다르다.

```python
# 파이썬 2
s = "hello"  # 바이트 시퀀스
u = u"안녕하세요"  # 유니코드 문자열

# 파이썬 3
s = "안녕하세요"  # 유니코드 문자열
b = b"hello"     # 바이트 시퀀스 (ASCII만 가능)

# 변환이 필요함
encoded = s.encode('utf-8')  # 문자열 → 바이트
decoded = b.decode('utf-8')  # 바이트 → 문자열
```

파이썬 2에서는 문자열(str)이 바이트 시퀀스였지만, 파이썬 3에서는 문자열은 유니코드 문자의 시퀀스이고 바이트는 별도의 bytes 타입으로 분리되었다.

## A.5 유니코드 처리 개선

파이썬 3에서는 문자열의 유니코드 처리가 더 일관되게 개선되었다.

```python
# 파이썬 2의 문제점
# -*- coding: utf-8 -*-  # 소스 코드 인코딩 선언 필요
s = "안녕"  # 기본 str은 바이트 시퀀스
u = u"안녕"  # 유니코드 문자열을 위해 u 접두사 필요

# 파이썬 3
s = "안녕"  # 기본적으로 유니코드 (u 접두사 불필요)
print(len(s))  # 2 (문자 개수로 계산)
b = s.encode('utf-8')
print(len(b))  # 6 (바이트 수로 계산)
```

파이썬 2에서는 유니코드 문자열을 다룰 때 많은 문제가 있었다. 파이썬 3에서는 모든 문자열이 기본적으로 유니코드이며 인코딩 처리가 더 일관적으로 개선되었다.

## A.6 range 동작 방식 변경

파이썬 2의 `range`와 `xrange`가 파이썬 3에서는 통합되었다.

```python
# 파이썬 2
numbers = range(10000000)  # 실제 리스트 생성 (메모리에 저장)
xnumbers = xrange(10000000)  # 이터레이터처럼 동작

# 파이썬 3
numbers = range(10000000)  # 범위 객체만 생성 (값은 필요할 때 생성)
print(type(numbers))  # <class 'range'>
```

파이썬 2의 range는 목록을 반환했지만, 파이썬 3의 range는 메모리 효율적인 이터레이터처럼 동작한다(파이썬 2의 xrange와 유사). 큰 범위를 생성할 때 메모리를 절약할 수 있다.

## A.7 예외 처리 문법 변경

파이썬 2와 3에서 예외 처리 문법이 다르다.

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

예외를 처리할 때 파이썬 3에서는 'as' 키워드 사용이 필수적으로 바뀌었고, 예외 객체의 관리 방식도 변경되었다.

## A.8 입력 함수 통합

파이썬 2의 `input`과 `raw_input`이 파이썬 3에서는 단일 함수로 통합되었다.

```python
# 파이썬 2
number = input("숫자 입력: ")  # "10"을 입력하면 정수 10으로 평가
name = raw_input("이름 입력: ")  # 문자열로 반환

# 파이썬 3
name = input("이름 입력: ")  # 항상 문자열로 반환
number = int(input("숫자 입력: "))  # 평가가 필요하면 명시적 변환 필요
```

파이썬 2에서는 `input()`이 입력을 평가(eval)하고, `raw_input()`이 문자열을 반환했지만, 파이썬 3에서는 `input()`이 항상 문자열을 반환하고 `raw_input()`은 삭제되었다.

## A.9 딕셔너리 뷰 객체 도입

파이썬 3에서는 딕셔너리의 메서드가 반환하는 값의 형태가 변경되었다.

```python
# 파이썬 2
d = {'a': 1, 'b': 2}
keys = d.keys()  # ['a', 'b'] (리스트)
d['c'] = 3
print(keys)  # ['a', 'b'] (딕셔너리 변경에 영향 없음)

# 파이썬 3
d = {'a': 1, 'b': 2}
keys = d.keys()  # dict_keys(['a', 'b']) (뷰 객체)
d['c'] = 3
print(keys)  # dict_keys(['a', 'b', 'c']) (딕셔너리 변경 반영)
```

파이썬 3에서는 딕셔너리의 `.keys()`, `.values()`, `.items()` 메서드가 리스트 대신 뷰 객체를 반환한다. 이 뷰 객체는 원본 딕셔너리의 변경 사항을 자동으로 반영한다.

## A.10 정렬 동작 변경

파이썬 3에서는 서로 다른 타입 간의 정렬 동작이 제한되었다.

```python
# 파이썬 2
sorted([1, 'a', 2, 'b'])  # 작동함

# 파이썬 3
# sorted([1, 'a', 2, 'b'])  # TypeError: '<' not supported between instances of 'str' and 'int'
```

파이썬 3에서는 다른 타입 간의 정렬이 불가능해졌고, 모든 요소가 상호 비교 가능해야만 정렬이 가능하다.

## A.11 이터레이터 반환 기본화

파이썬 3에서는 많은 내장 함수들이 리스트 대신 이터레이터를 반환하도록 변경되었다.

```python
# 파이썬 2
print(map(lambda x: x*2, [1, 2, 3]))  # [2, 4, 6] (리스트)

# 파이썬 3
print(map(lambda x: x*2, [1, 2, 3]))  # <map object at 0x...> (이터레이터)
print(list(map(lambda x: x*2, [1, 2, 3])))  # [2, 4, 6] (리스트로 변환 필요)
```

파이썬 3에서는 더 많은 함수와 메서드가 메모리 효율적인 이터레이터를 반환하도록 변경되었다. 이는 `map()`, `filter()`, `zip()` 등의 함수에 적용된다.

## A.12 호환성 팁 및 마이그레이션 도구

파이썬 2와 3 사이의 코드 이식성을 향상시킬 수 있는 방법과 도구들이 있다.

### a. 호환 코드 작성 패턴

두 버전에서 모두 작동하는 코드를 작성하기 위한 몇 가지 패턴:

```python
# 모듈 임포트 호환성
try:
    # 파이썬 3
    from functools import reduce
except ImportError:
    # 파이썬 2에서는 내장 함수
    pass

# 정수 나눗셈 일관성 유지
result = 7 // 5  # 두 버전 모두 정수 나눗셈 결과: 1

# 문자열/바이트 처리
def ensure_str(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s

# 입력 함수 일관성
try:
    # 파이썬 2
    input_func = raw_input
except NameError:
    # 파이썬 3
    input_func = input

name = input_func("이름을 입력하세요: ")
```

### b. six 라이브러리 활용

`six`는 파이썬 2와 3 간의 호환성 레이어를 제공하는 라이브러리이다:

```python
# 설치: pip install six
import six

# 파이썬 버전 확인
if six.PY2:
    # 파이썬 2 전용 코드
    print "파이썬 2"
else:
    # 파이썬 3 전용 코드
    print("파이썬 3")

# 문자열/바이트 처리
text = six.u("유니코드 문자열")  # 두 버전 모두 유니코드 문자열
binary = six.b("바이트 문자열")  # 두 버전 모두 바이트 문자열

# 이터레이터 함수
for item in six.iteritems({"a": 1, "b": 2}):
    print(item)  # 두 버전 모두 이터레이터 반환
```

### c. 2to3 자동 변환 도구

파이썬 표준 라이브러리에 포함된 `2to3`는 파이썬 2 코드를 파이썬 3 코드로 자동 변환해주는 도구이다:

```bash
# 기본 사용법
2to3 script.py         # 변환 내용 프리뷰
2to3 -w script.py      # 실제 파일 변환 (-w: write)
2to3 -w project_dir/   # 디렉토리 내 모든 파일 변환
```

주요 변환 처리:

- `print` 문을 함수로 변환
- 이터레이터 메서드 변환 (예: `.iteritems()` → `.items()`)
- 상대 임포트 문법 수정
- `except Exception, e:` → `except Exception as e:`
- `unicode`, `str` 타입 통합

### d. modernize와 futurize 도구

`python-modernize`와 `python-future` 패키지는 2to3를 기반으로 하지만, 파이썬 2와 3에서 모두 실행 가능한 코드를 생성하는 도구를 제공한다:

```bash
# 설치
pip install python-modernize
pip install future

# 사용법
python-modernize -w script.py
futurize -w script.py
```

python-future 사용 예시:

```python
# future 임포트
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()

# 이제 코드가 파이썬 2와 3에서 일관되게 동작
```

파이썬 2는 공식적으로 2020년 1월 1일부로 지원이 종료되었으므로, 새 프로젝트는 파이썬 3로 시작하는 것이 좋다. 하지만 기존 파이썬 2 코드를 유지보수하거나 이전해야 할 경우 위의 도구들이 유용하게 활용될 수 있다.

## A.13 파이썬 3.X 버전별 주요 기능

다음은 파이썬 3.0부터 주요 버전별로 추가된 중요 기능들의 요약이다. 이를 통해 파이썬 3 시리즈 내에서도 지속적으로 발전해온 주요 기능들을 한눈에 파악할 수 있다.

### 파이썬 3.0 (2008-12)

- 유니코드 기반 문자열
- `print()` 함수화
- `range()` 이터레이터 동작
- 정수 나눗셈 연산자 `/` 변경
- 예외 구문 변경(`as` 키워드)
- 뷰와 이터레이터 기반 `dict` 메서드

### 파이썬 3.1 (2009-06)

- `collections.OrderedDict` 추가
- 수 천 단위 구분자 `1_000_000` 지원
- 중첩된 `with` 구문
- `importlib` 패키지

### 파이썬 3.2 (2011-02)

- `argparse` 모듈 추가
- `functools.lru_cache` 추가
- `concurrent.futures` 모듈
- `functools.total_ordering` 데코레이터
- 문자열 `str.format_map()`

### 파이썬 3.3 (2012-09)

- `yield from` 구문 추가
- 새로운 `venv` 모듈
- `implicit namespace packages`
- `unittest.mock` 모듈 통합
- `ipaddress` 모듈

### 파이썬 3.4 (2014-03)

- `pathlib` 모듈
- `enum` 모듈
- `statistics` 모듈
- `asyncio` 모듈(비동기 프로그래밍)
- pickle 프로토콜 버전 4

### 파이썬 3.5 (2015-09)

- `async`/`await` 구문
- 행렬 곱셈 연산자 `@`
- 추가 언패킹 일반화 `*args`/`**kwargs`
- 타입 힌트 도입
- `zipapp` 모듈

### 파이썬 3.6 (2016-12)

- f-strings (`f"값: {변수}"`)
- 비동기 제너레이터
- 변수 타입 어노테이션
- 숫자 리터럴의 밑줄 구분자 `10_000`
- `secrets` 모듈

### 파이썬 3.7 (2018-06)

- 딕셔너리 순서 보장
- `breakpoint()` 내장 함수
- 데이터클래스(`@dataclass`)
- 타입 힌팅 지연 평가
- 컨텍스트 변수(`contextvars`)

### 파이썬 3.8 (2019-10)

- 왈러스 연산자 `:=`
- 위치 전용 매개변수(`/`)
- f-string `=` 지정자
- 멀티스레딩 성능 개선
- pickle 프로토콜 버전 5

### 파이썬 3.9 (2020-10)

- 딕셔너리 병합/업데이트 연산자(`|`, `|=`)
- 문자열 메서드 `removeprefix()`, `removesuffix()`
- 제네릭 타입으로 내장 컬렉션 사용
- 시간대 관련 개선사항
- 파서 개선(PEG)

### 파이썬 3.10 (2021-10)

- 구조적 패턴 매칭(`match`/`case`)
- 컨텍스트 관리자에서 괄호 생략
- 더 명확한 오류 메시지
- 제네릭의 타입 유니온 연산자(`|`)
- 중첩된 `ParamSpec`

### 파이썬 3.11 (2022-10)

- 성능 향상(10-60%)
- 예외 그룹
- 타입 변수 구문 간소화
- TOML 내장 지원(`tomllib`)
- 자체 타입 지원 향상

### 파이썬 3.12 (2023-10)

- 타입 힌팅 개선
- f-string 구문 단순화
- 새로운 파라미터 사양 변수
- 제로 저장 비용 C API
- 파이썬 하위 인터프리터 지원

이러한 발전 과정을 통해 파이썬은 계속해서 더 강력하고, 안전하며, 표현력이 풍부한 언어로 발전해 왔다. 특히 3.5 이후 버전부터는 비동기 프로그래밍, 타입 힌트, 데이터 처리 등에 중점을 두고 발전해 왔음을 알 수 있다.

---
> [목차로 돌아가기](../../README.md) | [이전: 시퀀스 타입과 슬라이싱](./1_9_sequence_types.md) | [다음: 파이썬 Pickle 객체 직렬화](./appendix_b_pickle.md)
