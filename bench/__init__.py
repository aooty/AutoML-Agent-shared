"""공개 데이터 벤치마크: 팔 세 개, 데이터셋 다섯 개, 판정 규칙 하나.

느슨한 스크립트 폴더가 아니라 패키지로 둔 이유는 여기 있는 것들이 모두 저장소 루트에서
``python -m bench.<module>``로 돌아야 하기 때문이다. ``automl_agent.scripts.train``과 같은
호출 방식이고, ``bench.datasets``와 ``automl_agent.dataset.features`` 양쪽 import가 함께
풀리는 유일한 방식이다. ``python bench/fetch.py``로 돌리면 인터프리터가 루트가 아니라 스크립트
디렉터리를 경로에 올리므로 앞쪽만 풀린다.

배포에는 들어가지 않는다 — ``[tool.setuptools.packages.find]``는 ``automl_agent*``만 잡는다.
이건 숫자를 검산하는 사람이 읽고 다시 돌릴 수 있어야 하는 저장소 도구이고, 설치 프로그램이
사용자 기계에 올리는 것과는 다른 물건이다.
"""
