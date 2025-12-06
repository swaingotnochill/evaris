# Evaris Server

FastAPI backend for running LLM-as-judge evaluations and storing results.

## Features

- REST API for running evaluations
- Multi-tenant support with Row Level Security (RLS)
- Prisma ORM with PostgreSQL
- JWT authentication for internal service communication
- Tracing and logging endpoints

## Development

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Generate Prisma client
prisma generate

# Run server
uvicorn evaris_server.app:create_app --factory --reload

# Run tests
pytest tests/
```

## Architecture

```
evaris-server/
  src/evaris_server/
    api/           # FastAPI routes and schemas
    db/            # Database client and models
    middleware/    # Auth middleware
    services/      # Business logic (evaluation runner)
  prisma/          # Prisma schema
  tests/           # Unit and integration tests
```
