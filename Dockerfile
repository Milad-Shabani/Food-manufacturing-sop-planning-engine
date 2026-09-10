FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["sh", "-c"]
CMD ["python -m sop_planning.cli generate-data --seed 42 && python -m sop_planning.cli run"]
