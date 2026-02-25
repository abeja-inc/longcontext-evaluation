ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Install uv
RUN apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

# Install dependencies (3rd party only)
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --format requirements-txt \
    --no-emit-workspace \
    -o /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# Install local workspace packages (editable)
COPY packages ./packages
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-deps \
    -e /workspace/packages/benchmarks \
    -e /workspace/packages/llm_inference

# Check imports (use the venv python explicitly)
ENTRYPOINT ["python3", "-c"]
CMD ["import benchmarks, llm_inference; print('IMPORT CHECK OK')"]
