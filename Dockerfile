FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
ARG PROPHITBET_COMMIT=49fa5ff14783af5dbbac7919ec743ab41628fcac
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgomp1 graphviz && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor.git /opt/prophitbet \
 && cd /opt/prophitbet && git checkout "$PROPHITBET_COMMIT" && rm -rf .git
WORKDIR /opt/prophitbet
COPY requirements-web.txt /opt/prophitbet/requirements-web.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements-web.txt
COPY web /opt/prophitbet/web
COPY deploy/entrypoint.sh /opt/deploy/entrypoint.sh
RUN chmod +x /opt/deploy/entrypoint.sh
ENV PYTHONPATH=/opt/prophitbet
EXPOSE 8080
CMD ["/opt/deploy/entrypoint.sh"]
