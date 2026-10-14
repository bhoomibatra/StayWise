.PHONY: setup train test app clean

# ─── Setup ────────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ─── Train models ─────────────────────────────────────────────────────────────
train:
	python -m src.train

# ─── Run tests ────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ─── Launch Streamlit app ─────────────────────────────────────────────────────
app:
	streamlit run app.py

# ─── Clean generated artifacts ───────────────────────────────────────────────
clean:
	del /Q models\*.joblib 2>nul || true
	del /Q reports\*.csv reports\*.png 2>nul || true
	del /Q data\processed\*.csv 2>nul || true
