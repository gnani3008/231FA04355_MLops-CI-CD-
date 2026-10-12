# Breast Cancer Classification: Feast + MLflow + Model Registry

- `train_model.py`: trains six models, logs MLflow runs, registers Version 1 and Version 2
- `test_model.py`: pytest tests
- `requirements.txt`: dependencies
- `.github/workflows/cicd.yml`: CI/CD workflow (tests + training on every push)
- `index.html`: project page (enable GitHub Pages: Settings > Pages > Deploy from branch > main / root)

Run locally:

    pip install -r requirements.txt
    pytest -v
    python train_model.py
    mlflow ui
