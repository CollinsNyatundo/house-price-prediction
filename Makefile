.PHONY: clean lint format test coverage run docker-build docker-run help

# Python interpreter
PYTHON = python

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  clean      - Remove build artifacts and cache files"
	@echo "  lint       - Run linters (flake8)"
	@echo "  format     - Format code with black and isort"
	@echo "  test       - Run tests"
	@echo "  coverage   - Run tests with coverage report"
	@echo "  run        - Run the Streamlit app"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 src tests app.py

format:
	black src tests app.py
	isort src tests app.py

test:
	pytest tests/

coverage:
	pytest --cov=src tests/ --cov-report=term --cov-report=html

run:
	streamlit run app.py

docker-build:
	docker build -t house-price-prediction .

docker-run:
	docker run -p 8501:8501 house-price-prediction 