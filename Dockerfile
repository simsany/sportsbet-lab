FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:99

ARG PROPHITBET_COMMIT=49fa5ff14783af5dbbac7919ec743ab41628fcac

# --------------------------------------------------
# SYSTEM DEPENDENCIES
# --------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    graphviz \
    ca-certificates \
    libgl1 \
    libglx0 \
    libegl1 \
    libopengl0 \
    libdbus-1-3 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libsm6 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxcb-util1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb-xinput0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# ORIGINAL PROPHITBET SOURCE
# --------------------------------------------------

RUN git clone \
    https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor.git \
    /opt/prophitbet \
    && cd /opt/prophitbet \
    && git checkout "$PROPHITBET_COMMIT" \
    && rm -rf .git

WORKDIR /opt/prophitbet

# --------------------------------------------------
# PYTHON TOOLING
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    --upgrade pip setuptools wheel

# --------------------------------------------------
# GUI AND DATA PROCESSING
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    PySide6==6.11.2 \
    numpy==1.26.4 \
    pandas==3.0.5 \
    pyarrow==25.0.1 \
    scipy==1.17.1 \
    openpyxl==3.1.5 \
    requests==2.34.2 \
    lxml==6.1.1 \
    tqdm==4.70.0 \
    fuzzywuzzy==0.18.0 \
    python-Levenshtein==0.27.3

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    matplotlib==3.11.1 \
    seaborn==0.13.2 \
    dcor==0.7 \
    Boruta==0.4.3 \
    dtreeviz==2.3.2 \
    graphviz==0.21

# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    scikit-learn==1.9.0 \
    imbalanced-learn==0.14.2 \
    xgboost==3.2.0 \
    lightgbm==4.7.0 \
    tensorflow==2.15.1 \
    shap==0.49.1

# --------------------------------------------------
# OPTIMIZATION AND DATABASE
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    optuna==4.9.0 \
    optuna-dashboard==0.20.0 \
    SQLAlchemy==2.0.51

# --------------------------------------------------
# PYTORCH RUNTIME DEPENDENCIES
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    filelock \
    typing-extensions \
    sympy \
    networkx \
    jinja2 \
    fsspec

# --------------------------------------------------
# CPU-ONLY PYTORCH
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0

# --------------------------------------------------
# REINFORCEMENT LEARNING
# --------------------------------------------------

RUN python -m pip install --no-cache-dir --no-compile \
    gymnasium==1.3.0 \
    stable-baselines3==2.9.0 \
    sb3-contrib==2.9.0

# --------------------------------------------------
# VERIFY PYTHON AND PYTORCH
# --------------------------------------------------

RUN python -c "\
import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available())"

# --------------------------------------------------
# VERIFY QT / XCB PLUGIN DEPENDENCIES
# --------------------------------------------------

RUN python - <<'PY'
import os
import subprocess
from pathlib import Path
from PySide6.QtCore import QLibraryInfo

plugins = Path(
    QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
) / "platforms"

plugin = plugins / "libqxcb.so"

print("Qt plugin directory:", plugins)
print("Qt XCB plugin:", plugin)

if not plugin.is_file():
    raise RuntimeError(f"Qt XCB plugin not found: {plugin}")

result = subprocess.run(
    ["ldd", str(plugin)],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.returncode != 0:
    raise RuntimeError("Could not inspect Qt XCB plugin dependencies")

if "not found" in result.stdout:
    raise RuntimeError("Missing Qt XCB shared library dependency")

print("Qt XCB dependencies verified successfully")
PY

# --------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------

COPY deploy/entrypoint.sh /opt/deploy/entrypoint.sh

RUN chmod +x /opt/deploy/entrypoint.sh

EXPOSE 8080

CMD ["/opt/deploy/entrypoint.sh"]
