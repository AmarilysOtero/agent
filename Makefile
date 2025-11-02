install:
	python -m pip install -r requirements.txt

lint:
	ruff check src tests
	mypy src

format:
	black src tests

test:
	pytest -q

run:
	python src/news_reporter/app.py
