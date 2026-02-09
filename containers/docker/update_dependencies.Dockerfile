ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Install uv
RUN apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

# Bring in your project definition (needed for uv lock)
COPY pyproject.toml ./
# 既に uv.lock があるなら持ち込んでもOK（無くてもOK）
# COPY uv.lock ./

# Freeze base dependencies -> constraints
RUN python3 -m pip freeze --all > /workspace/base-image-constraints.txt

ENTRYPOINT ["/bin/bash"]
CMD ["bash"]
