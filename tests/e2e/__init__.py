"""End-to-end tests for evaris.

These tests verify the full integration flow:
- Python SDK -> evaris-web -> evaris-server -> PostgreSQL

Requirements:
- Docker with docker-compose for running services
- Or manually running evaris-server and PostgreSQL

Run with: pytest tests/e2e -v --e2e
"""
