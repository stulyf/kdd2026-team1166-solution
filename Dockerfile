FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ripgrep ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

COPY . .

ENTRYPOINT ["sh", "-c", "mkdir -p /logs && uv run dabench \"$@\" 2>&1 | tee /logs/runtime.log", "--"]
CMD ["run-benchmark", "--config", "configs/submission.yaml"]
