# syntax=docker/dockerfile:1
# One image for the web app (`serve`) and the pipeline jobs (`update`, `backfill`, ...).

FROM python:3.12.16-slim-trixie AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
# Dependencies first so code edits do not invalidate the dependency layer.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps --no-cache-dir . && python -c "import cs_paper_radar.web.app"

FROM python:3.12.16-slim-trixie
ARG UID=10001
RUN groupadd --gid ${UID} app \
    && useradd --uid ${UID} --gid ${UID} --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown ${UID}:${UID} /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp \
    DATA_DIR=/data MODEL_DIR=/data/models HF_HOME=/data/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
USER ${UID}:${UID}
WORKDIR /tmp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status != 200)"
ENTRYPOINT ["cs-paper-radar"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
