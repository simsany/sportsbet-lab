FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=xcb DISPLAY=:99 QT_X11_NO_MITSHM=1
ARG PROPHITBET_COMMIT=49fa5ff14783af5dbbac7919ec743ab41628fcac
RUN apt-get update && apt-get install -y --no-install-recommends git xvfb x11vnc novnc websockify graphviz libgl1 libegl1 libxkbcommon0 libdbus-1-3 libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxrender1 libxext6 libsm6 libfontconfig1 libfreetype6 libglib2.0-0 libx11-6 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor.git /opt/prophitbet && cd /opt/prophitbet && git checkout "$PROPHITBET_COMMIT"
WORKDIR /opt/prophitbet
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && python -m pip install --no-cache-dir -r requirements.txt
COPY deploy/entrypoint.sh /opt/deploy/entrypoint.sh
RUN chmod +x /opt/deploy/entrypoint.sh
EXPOSE 8080
CMD ["/opt/deploy/entrypoint.sh"]
