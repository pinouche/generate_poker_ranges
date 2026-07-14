# The preflop API only reads text charts: no solver, no matplotlib, no solve output.
# Copy just what it needs and the image stays small and quick to rebuild.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Keep the repo's layout: preflop_advisor resolves the charts relative to its own file
# (../../ranges/...), so a flattened copy would not find them.
COPY resources/python/preflop_advisor.py resources/python/api.py resources/python/
COPY ["ranges/qb_ranges", "ranges/qb_ranges/"]

EXPOSE 8000

# One worker: the charts are small text files read per request, so this is IO-trivial and
# the container is meant to sit next to one table, not serve a fleet.
CMD ["uvicorn", "api:app", "--app-dir", "resources/python", "--host", "0.0.0.0", "--port", "8000"]
