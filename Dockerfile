
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_QPA_PLATFORM=xcb \
    DISPLAY=:99 \
    QT_X11_NO_MITSHM=1

ARG PROPHITBET_COMMIT=49fa5ff14783af5dbbac7919ec743ab41628fcac

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    graphviz \
    libgl1 \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libfontconfig1 \
    libfreetype6 \
    libglib2.0-0 \
    libx11-6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone original ProphitBet source
RUN git clone https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor.git /opt/prophitbet \
    && cd /opt/prophitbet \
    && git checkout "$PROPHITBET_COMMIT" \
    && rm -rf .git

WORKDIR /opt/prophitbet

# Upgrade pip tooling
RUN python -m pip install --no-cache-dir --no-compile \
    --upgrade pip setuptools wheel

# 1. GUI and data processing
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

# 2. Visualization
RUN python -m pip install --no-cache-dir --no-compile \
    matplotlib==3.11.1 \
    seaborn==0.13.2 \
    dcor==0.7 \
    Boruta==0.4.3 \
    dtreeviz==2.3.2 \
    graphviz==0.21

# 3. Machine learning
RUN python -m pip install --no-cache-dir --no-compile \
    scikit-learn==1.9.0 \
    imbalanced-learn==0.14.2 \
    xgboost==3.2.0 \
    lightgbm==4.7.0 \
    tensorflow==2.15.1 \
    shap==0.49.1

# 4. Optimization and database
RUN python -m pip install --no-cache-dir --no-compile \
    optuna==4.9.0 \
    optuna-dashboard==0.20.0 \
    SQLAlchemy==2.0.51

# 5. Install PyTorch runtime dependencies from PyPI
RUN python -m pip install --no-cache-dir --no-compile \
    filelock \
    typing-extensions \
    sympy \
    networkx \
    jinja2 \
    fsspec

# 6. CPU-only PyTorch
# Prevent installation of CUDA/NVIDIA packages
RUN python -m pip install --no-cache-dir --no-compile \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0

# 7. Reinforcement learning
RUN python -m pip install --no-cache-dir --no-compile \
    gymnasium==1.3.0 \
    stable-baselines3==2.9.0 \
    sb3-contrib==2.9.0

# Verify CPU-only PyTorch installation
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Entrypoint
COPY deploy/entrypoint.sh /opt/deploy/entrypoint.sh

RUN chmod +x /opt/deploy/entrypoint.sh

EXPOSE 8080

CMD ["/opt/deploy/entrypoint.sh"]
