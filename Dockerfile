# The advisor API reads text charts (preflop) and, when they are present, the solve
# jsons (flop/turn). The solves are tens of GB so they are never baked in -- mount them
# (-v $PWD/resources/outputs:/app/resources/outputs:ro) and postflop lights up; without
# the mount the container still answers preflop and 422s the rest.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Keep the repo's layout: the advisors resolve the charts and solves relative to their
# own file (../../ranges/..., ../outputs/...), so a flattened copy would not find them.
COPY resources/python/preflop_advisor.py resources/python/postflop_advisor.py \
     resources/python/api.py resources/python/
COPY ["ranges/qb_ranges", "ranges/qb_ranges/"]

EXPOSE 8000

# One worker: the charts are small text files read per request, so this is IO-trivial and
# the container is meant to sit next to one table, not serve a fleet.
CMD ["uvicorn", "api:app", "--app-dir", "resources/python", "--host", "0.0.0.0", "--port", "8000"]
