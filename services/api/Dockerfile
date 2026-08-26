FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
RUN useradd --create-home --uid 10001 appuser
COPY packages/portfolio_core /workspace/packages/portfolio_core
COPY services/api /workspace/services/api
COPY scripts /workspace/scripts
COPY data/demo /workspace/data/demo
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir /workspace/packages/portfolio_core /workspace/services/api pytest ruff
RUN mkdir -p /workspace/data /workspace/models && chown -R appuser:appuser /workspace
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "cd /workspace/services/api && alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
