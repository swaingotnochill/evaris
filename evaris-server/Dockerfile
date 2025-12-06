# Evaris Server Dockerfile
#
# Multi-stage build for a lean production image.
# Uses UV for fast, deterministic dependency management.
# Includes Node.js for schema management via @evarisai/schema.
#
# Usage:
#   docker build -t evaris-server .
#   docker run -p 8080:8080 --env-file .env evaris-server

# ==============================================================================
# Stage 1: Schema Builder (Node.js for npm package)
# ==============================================================================
FROM node:20-slim as schema-builder

WORKDIR /app

# Build arg for NPM authentication
ARG PERSONAL_NPM_TOKEN

# Copy npm configuration and package files
COPY package.json .

# Create .npmrc with GitHub Package Registry auth
RUN echo "@evarisai:registry=https://npm.pkg.github.com" > .npmrc && \
    echo "//npm.pkg.github.com/:_authToken=${PERSONAL_NPM_TOKEN}" >> .npmrc

# Install npm dependencies (fetches @evarisai/schema)
RUN npm install

# The schema is now in node_modules/@evarisai/schema/prisma/schema.prisma

# ==============================================================================
# Stage 2: Python Builder
# ==============================================================================
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml .

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies (without dev)
RUN uv pip install .

# Copy prisma directory structure
COPY prisma/ prisma/

# Copy schema from schema-builder stage
COPY --from=schema-builder /app/node_modules/@evarisai/schema/prisma/schema.prisma prisma/schema.prisma.from-npm

# Use a script to merge/update the schema (keep Python generator)
# For now, we'll use the local schema with Python generator already configured
# The npm schema serves as the source of truth for models

# Generate Prisma client
RUN prisma generate

# ==============================================================================
# Stage 3: Runtime
# ==============================================================================
FROM python:3.12-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r evaris && useradd -r -g evaris evaris

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY src/ src/
COPY prisma/ prisma/

# Set ownership
RUN chown -R evaris:evaris /app

USER evaris

# Environment defaults
ENV HOST="0.0.0.0"
ENV PORT="8080"
ENV ENVIRONMENT="production"
ENV DEBUG="false"
ENV LOG_LEVEL="INFO"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["uvicorn", "evaris_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
