ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /workspace

# Install uv
RUN apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

# Install dependencies
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Check imports (use the venv python explicitly)
ENTRYPOINT ["/opt/venv/bin/python", "-c"]
CMD ["import benchmarks, llm_inference; print('IMPORT CHECK OK')"]
