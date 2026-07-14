# One image, two entrypoints. The API and worker processes run the same codebase;
# Compose (or your orchestrator) picks the command.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY vds ./vds
RUN pip install --no-cache-dir -e .

EXPOSE 8000
# Default: the API. The worker service overrides this with `autodataforge worker`.
CMD ["autodataforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
