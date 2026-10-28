.PHONY: help install test lint format clean run docker-build docker-run

help:
	@echo "Hive Mind Swarm System - Available Commands"
	@echo ""
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean cache files"
	@echo "  make run           - Run the system"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest hive/tests/ -v

test-unit:
	pytest hive/tests/test_v21_components.py -v

test-integration:
	pytest hive/tests/test_integration_v21.py -v

lint:
	flake8 hive/ --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	black hive/
	isort hive/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

run:
	python main.py

dashboard:
	python web_dashboard.py

docker-build:
	docker build -t hive-mind:latest .

docker-run:
	docker run -it hive-mind:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down
