ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Install uv
RUN apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

# Install dependencies
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
