FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Set venv location outside /app to prevent volume mount overwrites
ENV UV_PROJECT_ENVIRONMENT=/venv

# Install dependencies
# --frozen: ensure lockfile is respected
# --no-dev: don't install dev dependencies
RUN uv sync --frozen --no-dev

# Add virtual env to PATH
ENV PATH="/venv/bin:$PATH"

COPY . /app

EXPOSE 5000

# Flask env vars
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Use the python in the venv (which is now in PATH)
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--reload"]
