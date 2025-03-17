# 부록 C: 파이썬 모듈 시스템

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 Pickle 객체 직렬화](./appendix_b_pickle.md)

## C.1 파이썬 모듈의 기본 개념

__파이썬 모듈은 코드를 구조화하고 재사용하기 위한 메커니즘으로, `.py` 확장자를 가진 파일이 하나의 모듈이 된다.__ 모듈은 변수, 함수, 클래스 등을 포함할 수 있으며, 다른 스크립트에서 `import` 문을 사용해 가져올 수 있다.

파이썬의 모듈 시스템은 C/C++의 `#include` 또는 Java의 `import`와 유사하지만, 더 동적이고 유연한 특성을 가진다:

```python
# 기본적인 모듈 임포트
import math
print(math.sqrt(16))  # 4.0

# 특정 요소만 임포트
from random import randint
print(randint(1, 10))  # 1과 10 사이의 난수

# 별칭 사용
import numpy as np
print(np.array([1, 2, 3]))
```

## C.2 모듈 검색 경로와 동적 임포트

### a. 모듈 검색 경로 (import search path)

파이썬은 모듈을 임포트할 때 어디서 모듈을 찾을지 결정하기 위해 `sys.path` 리스트를 사용한다. 이는 C/C++의 포함 디렉토리 지정과 다르게, 프로그램 실행 중에 동적으로 변경될 수 있다:

```python
import sys
print(sys.path)  # 현재 모듈 검색 경로 출력

# 검색 경로의 일반적인 구성
# 1. 현재 디렉토리 또는 스크립트가 있는 디렉토리
# 2. PYTHONPATH 환경 변수에 지정된 디렉토리
# 3. 파이썬 표준 라이브러리 디렉토리
# 4. site-packages 디렉토리 (설치된 외부 패키지)
```

### b. 동적으로 경로 추가하기

C/C++과 달리, 파이썬은 실행 중에 모듈 검색 경로를 수정할 수 있다:

```python
import sys
import os

# 새 경로를 검색 경로에 추가
custom_module_path = os.path.join(os.path.dirname(__file__), 'custom_modules')
sys.path.append(custom_module_path)

# 이제 custom_modules 디렉토리의 모듈을 임포트할 수 있음
import my_custom_module
```

### c. 동적 모듈 검색 응용

검색 경로를 동적으로 조작하면 플러그인 시스템이나 확장 가능한 아키텍처를 구현할 수 있다:

```python
def load_plugins(plugin_dir):
    """특정 디렉토리의 모든 플러그인 모듈을 동적으로 로드"""
    import sys
    import os
    import importlib
    
    # 플러그인 디렉토리를 검색 경로에 추가
    sys.path.insert(0, plugin_dir)
    
    plugins = []
    for filename in os.listdir(plugin_dir):
        # .py 파일만 처리
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]  # .py 확장자 제거
            try:
                # 동적으로 모듈 임포트
                module = importlib.import_module(module_name)
                if hasattr(module, 'register_plugin'):
                    plugins.append(module.register_plugin())
            except ImportError as e:
                print(f"경고: {module_name} 로드 실패: {e}")
    
    return plugins
```

## C.3 블록 레벨 임포트와 동적 임포트

### a. 블록 레벨 임포트 (함수/조건문 내부 임포트)

C/C++과 달리, 파이썬은 모듈 단위가 아닌 임의의 블록 레벨에서 임포트를 수행할 수 있다:

```python
def process_image():
    # 함수 내부에서 임포트
    # 이 모듈은 함수가 호출될 때만 로드됨
    from PIL import Image
    
    # 나머지 이미지 처리 코드
    return Image.new('RGB', (100, 100))

if some_condition:
    # 조건부 임포트
    import special_module
    special_module.do_something()
```

블록 레벨 임포트의 주요 이점:

1. __지연 로딩__: 필요할 때만 모듈을 임포트하여 초기 로딩 시간 단축
2. __의존성 격리__: 특정 함수나 조건에서만 필요한 모듈을 해당 스코프에서만 사용
3. __순환 참조 방지__: 전역 수준의 순환 참조 회피
4. __선택적 기능__: 특정 환경이나 조건에서만 필요한 모듈 처리

### b. 실행 중 임포트 (런타임 임포트)

모듈 이름을 동적으로 결정하여 임포트할 수 있다:

```python
def import_strategy(strategy_name):
    """전략 이름에 따라 다른 모듈을 동적으로 임포트"""
    try:
        # __import__() 또는 importlib.import_module() 사용
        import importlib
        strategy_module = importlib.import_module(f"strategies.{strategy_name}")
        return strategy_module.Strategy()
    except ImportError:
        print(f"전략 '{strategy_name}'을 찾을 수 없습니다.")
        return None

# 사용 예시
user_strategy = "advanced"
strategy = import_strategy(user_strategy)
```

### c. 임포트 중 코드 실행

파이썬에서는 모듈이 임포트될 때 최상위 레벨의 코드가 실행된다. 이는 C/C++의 헤더 파일과는 매우 다른 동작이다:

```python
# my_module.py
print("my_module이 임포트되었습니다!")

x = 10
def foo():
    return x * 2
```

```python
# main.py
import my_module  # "my_module이 임포트되었습니다!" 출력됨
```

모듈 실행 코드와 임포트용 코드를 구분하기 위한 관용적 패턴:

```python
# module_with_main.py
def main_function():
    print("모듈이 스크립트로 실행되었습니다.")

if __name__ == "__main__":
    main_function()
else:
    print("모듈이 임포트되었습니다.")
```

이 패턴에서 `__name__` 변수는 모듈이 어떻게 실행되었는지에 따라 다른 값을 갖는다:

1. __직접 실행 시__: Python 인터프리터로 직접 파일을 실행할 때 (예: `python module_with_main.py`), `__name__`은 `"__main__"`으로 설정된다.
2. __임포트 시__: 다른 모듈에서 임포트할 때 (예: `import module_with_main`), `__name__`은 모듈의 실제 이름인 `"module_with_main"`으로 설정된다.

이를 활용하면 같은 파이썬 파일을 라이브러리로도 사용하고, 독립 실행 스크립트로도 사용할 수 있다:

```python
# 예시: 계산기 모듈
# calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# 이 부분은 모듈이 직접 실행될 때만 실행됨
if __name__ == "__main__":
    # 모듈 테스트 또는 명령줄 인터페이스로 활용
    print("계산기 모듈을 직접 실행했습니다.")
    x, y = 10, 5
    print(f"{x} + {y} = {add(x, y)}")
    print(f"{x} - {y} = {subtract(x, y)}")
```

이제 이 파일은 다음과 같이 활용할 수 있다:

- 라이브러리로 사용: `import calculator`로 임포트 시 함수만 사용 가능
- 직접 실행: `python calculator.py` 명령으로 실행 시 테스트 코드 실행

이러한 패턴은 파이썬 코드의 재사용성과 모듈성을 높이는 데 매우 유용하다.

## C.4 패키지 시스템

### a. 패키지 기본 구조

패키지는 모듈을 계층적으로 구성하는 디렉토리로, 반드시 `__init__.py` 파일을 포함해야 한다 (Python 3.3+에서는 선택적):

```plaintext
my_package/
├── __init__.py
├── module1.py
├── module2.py
└── subpackage/
    ├── __init__.py
    └── submodule1.py
```

### b. 패키지 임포트

패키지를 임포트할 때는 디렉토리 구조를 반영하여 점(.) 표기법을 사용한다:

```python
# 패키지 전체 임포트
import my_package

# 패키지 내 특정 모듈 임포트
from my_package import module1

# 서브패키지 내 모듈 임포트
from my_package.subpackage import submodule1
```

### c. `__init__.py` 파일의 역할

`__init__.py` 파일은 패키지를 초기화하는 데 사용되며, 패키지가 임포트될 때 실행된다. 이 파일을 통해 패키지 수준의 변수, 함수, 클래스 등을 정의할 수 있다:

```python
# my_package/__init__.py
print("my_package가 임포트되었습니다!")

# 패키지 수준 변수
package_variable = "Hello, Package!"

# 패키지 수준 함수
def package_function():
    print("This is a package function.")
```

### d. 네임스페이스 패키지

Python 3.3부터는 `__init__.py` 없이도 여러 디렉터리에 걸친 패키지 구성이 가능하다:

```python
# 여러 디렉터리에 걸쳐 존재하는 네임스페이스 패키지
# site-packages/ns_pkg/modA.py
# another-path/ns_pkg/modB.py

# Python은 두 위치를 모두 검색하여 통합된 ns_pkg 패키지로 제공
import ns_pkg.modA
import ns_pkg.modB  # 다른 디렉터리지만 같은 패키지로 임포트 가능
```

## C.5 모듈 언로딩

파이썬은 기본적으로 모듈을 언로드하는 기능을 제공하지 않지만, `sys.modules`를 조작하여 모듈을 언로드할 수 있다. 모듈을 언로드하면 메모리를 절약하고, 모듈의 변경사항을 반영할 수 있다:

```python
import sys

def unload_module(module_name):
    """모듈 언로드"""
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"{module_name} 모듈이 언로드되었습니다.")
    else:
        print(f"{module_name} 모듈이 로드되어 있지 않습니다.")

# 사용 예시
import math
unload_module('math')
```

### a. 모듈 언로딩의 내부 동작

파이썬에서 모듈 언로딩은 다음과 같은 과정으로 이루어진다:

```python
# 모듈 언로딩 과정 이해하기
import sys
import math

# 1. 모듈은 sys.modules 딕셔너리에 캐싱된다
print('math' in sys.modules)  # True

# 2. 모듈 언로딩 - sys.modules에서 제거
del sys.modules['math']
print('math' in sys.modules)  # False

# 3. 새로 임포트하면 다시 로드된다
import math  # 모듈을 새로 로드함
print('math' in sys.modules)  # True
```

### b. 완전한 언로딩을 위한 참조 제거

모듈을 `sys.modules`에서 제거해도, 다른 곳에서 해당 모듈을 참조하고 있다면 가비지 컬렉션이 되지 않는다. 완전한 언로딩을 위해서는 모든 참조를 제거해야 한다:

```python
# 완전한 모듈 언로딩
import sys
import my_module  # 예시 모듈

# 로컬 네임스페이스에서 참조 제거
if 'my_module' in locals():
    del my_module

# 글로벌 네임스페이스에서 참조 제거
if 'my_module' in globals():
    del globals()['my_module']

# 캐시에서 제거
if 'my_module' in sys.modules:
    del sys.modules['my_module']

# 이제 my_module은 완전히 언로드되었으며, 다음 임포트 시 새로 로드됨
```

### c. 모듈 언로딩 활용 사례

모듈 언로딩은 다음과 같은 상황에서 유용하게 사용된다:

1. __개발 중 코드 변경 반영__: 모듈 코드 수정 후 다시 로드하여 변경사항 테스트
2. __메모리 관리__: 대용량 모듈을 더 이상 사용하지 않을 때 메모리 확보
3. __충돌 해결__: 모듈 간 충돌이나 문제가 있을 때 특정 모듈을 다시 로드
4. __동적 모듈 교체__: 런타임 중에 다른 구현으로 모듈 교체

```python
# 개발 중 코드 변경사항 반영 예시
import sys
import importlib
import my_module

# 모듈 코드 수정 후...

# 방법 1: 언로드 후 다시 임포트
if 'my_module' in sys.modules:
    del sys.modules['my_module']
import my_module  # 변경된 코드로 새로 로드

# 방법 2: importlib.reload 사용 (Python 3.4+)
importlib.reload(my_module)  # 모듈 재로드 (언로드 없이)
```

### d. 모듈 언로딩의 한계

모듈 언로딩에는 몇 가지 중요한 한계와 주의사항이 있다:

1. __완전한 언로딩이 어려움__: 모듈이 다른 모듈에 의해 간접적으로 참조될 수 있음
2. __상태 유지 문제__: 모듈이 전역 상태를 변경했을 경우, 해당 변경은 유지됨
3. __사이드 이펙트__: 모듈이 로드될 때 사이드 이펙트(파일 열기, 리소스 할당 등)가 있다면, 언로딩 시 자동으로 정리되지 않음
4. __싱글톤 객체 재설정__: 모듈 내 싱글톤 객체는 언로드 후 재로드 시 새로 생성됨

```python
# 한계 예시: 다른 모듈을 통한 간접 참조
# moduleA.py
import moduleB

# 메인 코드
import moduleA
import sys

# moduleB는 직접 임포트하지 않았지만 sys.modules에 있음
print('moduleB' in sys.modules)  # True

# moduleA를 언로드해도 moduleB는 그대로 남음
del sys.modules['moduleA']
print('moduleB' in sys.modules)  # True
```

## C.6 파이썬 모듈 시스템의 고유 특징

### a. 모듈 재사용 (싱글턴 패턴)

파이썬에서는 같은 모듈을 여러 번 임포트해도 단 한 번만 실행된다:

```python
# first.py
import math
print("math 모듈 ID:", id(math))

# second.py
import math
print("math 모듈 ID:", id(math))

# main.py
import first
import second
# 두 출력의 ID는 동일함 - 동일한 객체를 참조
```

### b. 모듈 다시 로드하기

모듈이 이미 로드되었더라도 강제로 다시 로드할 수 있다:

```python
import importlib
import my_module

# 모듈 수정 후 다시 로드
importlib.reload(my_module)
```

### c. `__import__` 함수와 `importlib`

저수준 임포트 제어를 위한 도구:

```python
# __import__ 함수 (저수준)
math_module = __import__('math')
print(math_module.sqrt(16))  # 4.0

# importlib 모듈 (권장)
import importlib
random_module = importlib.import_module('random')
print(random_module.randint(1, 10))
```

### d. 모듈의 특수 변수들

파이썬의 모든 모듈은 특수 변수들을 자동으로 가진다:

```python
# 현재 모듈의 이름
print(__name__)  # '__main__' 또는 모듈 이름

# 현재 모듈의 파일 경로
print(__file__)  # '/path/to/current_file.py'

# 현재 모듈의 문서화 문자열
print(__doc__)   # 모듈 시작 부분의 문자열 리터럴

# 모듈이 정의한 공개 이름들 (from * 사용 시)
__all__ = ['func1', 'Class1']  # 명시적으로 설정 가능
```

## C.7 C/C++의 #include와 파이썬 import의 주요 차이점

파이썬의 `import`와 C/C++의 `#include`는 중요한 차이점이 있다:

| 측면 | C/C++ `#include` | 파이썬 `import` |
|------|---------------|--------------|
| 처리 시점 | 전처리 시간 (컴파일 전) | 런타임 (실행 중) |
| 내용 처리 | 텍스트 포함 (복사-붙여넣기) | 모듈 객체 생성 및 참조 |
| 중복 포함 | 가능 (보호 장치 필요) | 자동 방지 (한 번만 실행) |
| 경로 변경 | 컴파일 플래그로 고정 | 런타임에 동적 변경 가능 |
| 블록 레벨 | 불가능 (파일 수준만) | 가능 (함수/조건문 내 등) |
| 코드 실행 | 선언만 포함 (정의 제외) | 모듈 코드 실행됨 |
| 조건부 | 전처리 지시문 (#ifdef 등) | 일반 파이썬 코드 (if 문) |

```python
# 파이썬에서 조건부 임포트
import platform

if platform.system() == 'Windows':
    import winreg  # Windows 레지스트리 모듈
else:
    import pwd     # UNIX 사용자 정보 모듈
```

## C.8 모범 사례

### a. 임포트 구성 및 정렬

```python
# 권장 임포트 순서
# 1. 표준 라이브러리
import os
import sys

# 2. 서드파티 패키지/모듈
import numpy as np
import pandas as pd

# 3. 로컬 애플리케이션/라이브러리
from myapp import utils
from myapp.models import User
```

### b. 효율적인 임포트

```python
# 필요한 것만 임포트 (권장)
from math import sqrt, log
result = sqrt(10) + log(10)

# 전체 모듈 임포트 (식별자 명확성)
import math
result = math.sqrt(10) + math.log(10)

# 와일드카드 임포트 (권장하지 않음)
from math import *  # 네임스페이스 오염 위험
result = sqrt(10) + log(10)
```

### c. 순환 참조 해결

```python
# a.py
def a_function():
    pass

def use_b():
    # 순환 참조 방지를 위한 지연 임포트
    from b import b_function
    b_function()

# b.py
from a import a_function  # a_function만 임포트

def b_function():
    a_function()
```

### d. 모듈 구성 패턴

```python
"""모듈 예시: my_module.py

이 모듈은 좋은 모듈 구성 패턴을 보여줍니다.
"""

# 표준 라이브러리
import os
import sys

# 서드파티 라이브러리
import numpy as np

# 상수 정의
MAX_SIZE = 100
DEFAULT_COLOR = 'red'

# 함수 정의
def process_data(data):
    """데이터를 처리합니다."""
    return data * 2

# 클래스 정의
class DataProcessor:
    """데이터 처리 클래스"""
    def __init__(self):
        self.data = []
    
    def add(self, item):
        self.data.append(item)

# 내부용 함수/변수 (관례적으로 _로 시작)
_internal_cache = {}
def _calculate_internal(x):
    return x * x

# 조건부 코드
if __name__ == "__main__":
    # 모듈이 직접 실행될 때만 실행됨
    print("모듈을 직접 실행했습니다.")
    processor = DataProcessor()
    processor.add(5)
```

## C.9 일반적인 문제와 해결책

### a. `ModuleNotFoundError` 해결

```python
# 오류: ModuleNotFoundError: No module named 'my_module'
# 해결책 1: PYTHONPATH 설정
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 해결책 2: setuptools와 pip 사용
# setup.py 작성 후 pip install -e . 실행

# 해결책 3: .pth 파일 사용
# site-packages에 custom.pth 파일 생성하여 경로 추가
```

### b. 임포트 순서에 따른 문제 (`ImportError`)

```python
# 잘못된 예 - 순환 참조
# main.py
import a
import b

# a.py
import b
def a_func():
    b.b_func()

# b.py
import a
def b_func():
    a.a_func()  # 이때 a 모듈이 완전히 초기화되지 않아 오류 발생 가능
```

### c. 환경별 조건부 임포트

```python
# 다양한 환경에서의 조건부 임포트
try:
    # 주 구현체 시도
    import preferred_library as lib
except ImportError:
    try:
        # 대체 구현체 시도
        import alternative_library as lib
    except ImportError:
        # 기본 구현체 사용
        import basic_implementation as lib

# 특정 플랫폼에 따른 임포트
import platform
if platform.system() == 'Windows':
    from .windows_impl import WindowsImplementation as Impl
elif platform.system() == 'Darwin':  # macOS
    from .macos_impl import MacOSImplementation as Impl
else:  # Linux 등
    from .linux_impl import LinuxImplementation as Impl
```

## C.10 파이썬 모듈 생명주기 요약

파이썬 모듈의 전체 생명주기를 이해하면 효율적인 코드 구성과 문제 해결에 도움이 된다:

### a. 모듈 검색 및 로딩

1. `import` 문 실행
2. `sys.modules` 캐시 확인
   - 캐시에 있음 → 캐시에서 모듈 객체 반환
   - 캐시에 없음 → 모듈 검색 실행
3. `sys.path` 경로 순차 검색
   - 모듈 찾음 → 모듈 코드 컴파일 및 실행
   - 모듈 못 찾음 → `ImportError` 발생
4. 모듈 객체 생성 및 `sys.modules`에 저장
5. 모듈 객체를 임포트 요청 네임스페이스에 바인딩

### b. 모듈의 메모리 관리와 언로딩

1. 모듈은 `sys.modules` 딕셔너리에 캐싱됨
2. 모듈 객체는 일반 객체와 같은 참조 카운팅으로 관리
3. 언로딩 과정:
   - `sys.modules`에서 제거 (`del sys.modules['모듈명']`)
   - 글로벌, 로컬 네임스페이스에서 참조 제거
   - 참조 카운트가 `0`이 되면 가비지 컬렉터가 메모리 해제
4. 인터프리터 종료 시 모든 모듈이 자동으로 언로딩됨

언로딩 관련 중요 사항:

- 명시적 언로딩은 `del sys.modules['module_name']`으로 수행
- 완전한 언로딩을 위해서는 모든 참조를 제거해야 함
- 실제 메모리 해제는 가비지 컬렉터의 작업 스케줄에 따라 이루어짐
- 실무에서는 주로 개발 중 코드 변경사항을 테스트하거나 메모리 이슈 해결 시 사용

언로딩과 관련해서 먼저 자세히 기술한 [모듈 언로딩](#c5-모듈-언로딩) 섹션에서 더 많은 정보를 참고할 수 있다.

### c. 모듈 재로딩 시나리오

1. 개발 중 코드 변경 반영 → `importlib.reload()` 사용
2. 설정 변경 후 모듈 재초기화 → 캐시에서 제거 후 재임포트
3. 리소스 이슈 해결 → 필요 없는 큰 모듈 명시적 언로딩
4. 동적 플러그인 시스템 → 런타임에 모듈 교체

파이썬의 모듈 시스템은 프로그램 실행 중 동적으로 코드를 구성하고 로드하는 유연한 메커니즘을 제공한다. 이 동적인 특성은 다른 언어보다 더 큰 유연성을 제공하지만, 올바르게 이해하고 사용하는 것이 중요하다.

---
> [목차로 돌아가기](../../README.md) | [이전: 파이썬 Pickle 객체 직렬화](./appendix_b_pickle.md)
