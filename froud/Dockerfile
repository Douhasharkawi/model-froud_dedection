FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn catboost imbalanced-learn joblib numpy pandas
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]