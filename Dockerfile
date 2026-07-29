# PV Re-Orientation Prioritization Agent — container image.
#
# Build:
#   docker build -t pv-agent .
#
# Run the pipeline (API key passed at runtime, never baked in):
#   docker run -e ANTHROPIC_API_KEY=sk-... pv-agent
#
# Run the deterministic tests (no API key needed):
#   docker run pv-agent pytest
#
# Run the agent eval (needs the API key):
#   docker run -e ANTHROPIC_API_KEY=sk-... pv-agent python evals/eval_agent.py
#
# Stable, widely-available base (our code is 3.10+; 3.12 satisfies requires-python).
FROM python:3.12-slim

WORKDIR /app

# Copy the whole project — including data/pv_sites_sample.csv, which the pipeline
# locates relative to the package, and evals/ for the eval command.
COPY . /app

# Clean, non-editable install of the package with its declared dependencies
# (python-dotenv, pvlib, pgeocode, pydantic-ai, ...) plus the [dev] extra
# (pytest) so `docker run pv-agent pytest` works. Non-editable is more reliable
# here — the container never edits code, and a real install lands the package in
# site-packages, so `python -m pv_agent.run` imports with no path juggling.
RUN pip install --no-cache-dir ".[dev]"

# Point the package at the copied-in dataset. Needed because a non-editable
# install lives in site-packages and cannot find the repo-root data/ dir on its
# own; this makes the location explicit.
ENV PV_AGENT_CSV=/app/data/pv_sites_sample.csv

# The ANTHROPIC_API_KEY is supplied at runtime via `-e ANTHROPIC_API_KEY=...`,
# NEVER baked into the image.

# Default command: run the full pipeline.
CMD ["python", "-m", "pv_agent.run"]
