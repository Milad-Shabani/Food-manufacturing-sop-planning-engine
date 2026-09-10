.PHONY: install generate-data run test lint docker-run

install:
	pip install -e ".[dev]" 2>/dev/null || { pip install -r requirements-dev.txt && pip install -e .; }

generate-data:
	python -m sop_planning.cli generate-data --seed 42

run:
	python -m sop_planning.cli run

test:
	pytest -v --cov=sop_planning --cov-report=term-missing

lint:
	flake8 src tests --max-line-length=100 --extend-ignore=E203
	black --check src tests

docker-run:
	docker compose up --build planning
