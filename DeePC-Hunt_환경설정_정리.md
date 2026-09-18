# DeePC-Hunt + coco_project_2026 환경 세팅 정리

WSL2(Ubuntu 22.04) 위에서 `DeePC-Hunt`와 `coco_project_2026`(coco_rocket_lander)을 한 노트북(`rocket.ipynb`)에서 같이 돌리기 위한 환경 구성 과정과, 그 과정에서 실제로 걸렸던 함정들을 정리합니다. 파일 자체(코드) 수정 내용은 제외하고, **환경/설치 관련 부분만** 다룹니다.

## 1. 왜 이렇게 복잡해졌는가 — 핵심 원인 하나

`coco_project_2026`의 `pyproject.toml`이 `python = ">=3.13,<3.15"`로 못 박혀 있고, 그 안에 고정된 numpy(2.4.4)/pandas(3.0.2)/scipy(1.17.1)/cvxpy(1.8.2)/ipython(9.12.0)도 전부 Python 3.11~3.12 이상을 요구합니다. 반면 그동안 쓰던 venv는 Python 3.10.12였기 때문에, 이 프로젝트를 설치하려는 시점부터 사실상 새 환경이 필요했습니다. `DeePC-Hunt` 쪽은 `numpy>=1.25.2`, `cvxpylayers>=0.1`, `torch>=1.0`처럼 조건이 느슨해서 3.13에서도 문제없이 설치됩니다.

## 2. Python 3.13 설치 (Ubuntu 22.04는 기본 저장소에 3.13이 없음)

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev
python3.13 --version   # Python 3.13.x 확인
```

## 3. venv는 반드시 리눅스 네이티브 경로에 생성 — `/mnt/c/...` 금지

```bash
mkdir -p ~/ws
python3.13 -m venv ~/ws/venv313
source ~/ws/venv313/bin/activate
```

**주의할 점**: `venv`는 실행한 인터프리터의 버전 그대로 가상환경을 만듭니다 (`python3.10 -m venv`는 무조건 3.10 venv). 다른 버전이 필요하면 그 버전의 인터프리터를 먼저 시스템에 설치해야 합니다.

**왜 `/mnt/c`를 피해야 하는가**: WSL에서 `/mnt/c/...`는 Windows 디스크를 빌려온 경로(DrvFS)라서, pip이 컴파일된 바이너리 패키지(numpy, torch 등)를 업그레이드/재설치할 때 파일 동기화가 깨지는 경우가 실제로 발생했습니다 (이번에 numpy가 `numpy/lib/__init__.py`와 `_npyio_impl.py`의 버전이 서로 안 맞는 상태로 깨져서 `AttributeError: module 'numpy.lib' has no attribute '_datasource'`가 났던 사례). **venv(site-packages)는 반드시 `~/ws/...`처럼 리눅스 쪽에 두고, 프로젝트 소스 코드만 `/mnt/c/...`에 있어도 무방**합니다 (소스는 단순 읽기/쓰기라 문제 없음).

## 4. poetry로 coco_project_2026 설치

```bash
pip install --upgrade pip
pip install poetry
cd /mnt/c/Users/JYB/Downloads/coco_project_2026-master
poetry env info   # Path가 ~/ws/venv313을 가리키는지 꼭 확인
poetry install
```

- `poetry install`은 이미 활성화된 venv가 있으면 새로 만들지 않고 그걸 그대로 씁니다 — 단, `poetry env info`로 반드시 확인.
- `gymnasium[box2d]` 빌드 중 `swig` 관련 에러가 나면: `sudo apt install -y swig`

## 5. DeePC-Hunt를 같은 venv에 추가 설치

```bash
cd /mnt/c/Users/JYB/Downloads/DeePC-Hunt
pip install -e .
```

## 6. CUDA / torch 관련 — 가장 조심해야 할 부분

### 6-1. GPU가 Pascal 세대(TITAN Xp, sm_61)라면

- 최신 torch(예: `2.14.0+cu130`, CUDA 13.0 빌드)는 NVIDIA가 CUDA 12.8/13.x 툴킷에서 Maxwell/Pascal(sm_50~61) 커널 컴파일 자체를 제거해버려서 **아예 그 GPU용 커널이 wheel 안에 없습니다.** 드라이버 문제가 아니라 wheel에 코드가 없는 것이므로 재설치로만 해결됩니다.
- 해결: 같은 torch 버전을 **cu126** 빌드로 설치하면 Pascal 커널이 아직 남아있습니다.
  ```bash
  pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
  ```
- **주의**: PyTorch의 공식 RFC에 따르면 `cu126` wheel 라인은 **PyTorch 2.15부터 제거될 예정**입니다. 즉 이 조합(2.14.0 + cu126)이 Pascal GPU로 pip wheel을 쓸 수 있는 사실상 마지막 조합입니다. 이후 torch를 업그레이드하면 이 문제가 재발합니다.
- Python 3.13용 cu126 wheel이 실제로 배포되는지는 새 venv에서 직접 설치해보며 확인 필요 (버전/파이썬 조합에 따라 없을 수도 있음 → 그 경우 cu118/121/124 등으로 대체).

### 6-2. (가장 중요) 결국 GPU 자체를 포기하는 게 맞는 이유

DeePC-Hunt의 `DeePC` 컨트롤러는 `cvxpylayers`를 통해 매 스텝마다 작은 볼록 QP를 `Clarabel`/`ECOS` 솔버로 풉니다. 최신 `cvxpylayers`(1.2.0)에서 이 두 solve_method는 내부적으로 `diffcp`라는 **CPU/numpy 전용** 백엔드를 거치기 때문에, 아무리 텐서들을 GPU에 올려놔도 QP를 실제로 푸는 단계에서 `TypeError: can't convert cuda:0 device type tensor to numpy` 에러로 막힙니다. GPU 네이티브 백엔드(`CUCLARABEL`, `MPAX`, `MOREAU`)도 라이브러리에 있지만 별도 패키지 설치가 필요하고 DeePC-Hunt 코드가 그걸 쓰도록 되어 있지 않습니다.

문제 크기(state 6개, 입력 3개, horizon 10)도 매우 작아서 GPU 이득이 없으므로:

```python
device = 'cpu'   # rocket.ipynb에서 'cuda' if torch.cuda.is_available() else 'cpu' 대신
```

로 고정하는 것이 가장 깔끔한 해결책입니다. CUDA/드라이버/wheel 버전 문제를 통째로 피할 수 있습니다.

### 6-3. numpy/torch 설치 시 일반 주의사항

- `pip install`로 numpy/torch를 업그레이드할 때는 **완전 삭제 후 재설치**가 안전합니다 (`pip install --upgrade`로 제자리 교체하면 특히 `/mnt/c` 환경에서 파일이 어중간하게 섞일 수 있음).
  ```bash
  pip uninstall numpy -y
  pip install --no-cache-dir --force-reinstall numpy
  ```
- `pip check`로 의존성 충돌을 주기적으로 확인.
- `cvxpylayers` 최신판은 `cvxpy>=1.9.0`, `python>=3.11`을 요구 — `coco_project_2026`의 `cvxpy` 고정 버전(1.8.2)과 살짝 어긋나 보이지만, pyproject 제약이 `^1.8.2`(즉 `<2.0.0`이면 다 허용)라서 새로 resolve하면 자연스럽게 맞춰짐 (심각한 충돌 아님).

## 7. 최종 확인용 스니펫

```bash
source ~/ws/venv313/bin/activate
python -c "
import numpy as np, torch, gymnasium as gym
from deepc_hunt import DeePC, Trainer
import coco_rocket_lander
print('numpy', np.__version__)
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('gymnasium', gym.__version__)
print('ALL IMPORTS OK')
"
```

## 요약 체크리스트

- [ ] venv는 `~/ws/venv313`처럼 **리눅스 네이티브 경로**에 생성 (절대 `/mnt/c` 아래 X)
- [ ] Python 3.13은 deadsnakes PPA로 설치
- [ ] `coco_project_2026`은 poetry로, `DeePC-Hunt`는 pip -e로, **같은 venv**에 설치
- [ ] box2d 빌드 에러 나오면 `sudo apt install swig`
- [ ] torch는 Pascal GPU라면 `cu126` 빌드 고정 (단, 2.15부터 사라짐 — 임시방편)
- [ ] 하지만 결국 `device='cpu'`로 고정하는 게 이 프로젝트에는 정답 (QP 솔버가 CPU 전용이라 GPU 무의미)
- [ ] numpy/torch 재설치는 upgrade보다 uninstall 후 clean install
