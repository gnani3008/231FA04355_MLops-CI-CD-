import pytest

import train_model as tm


@pytest.fixture(scope="module")
def df():
    return tm.load_data()


def test_dataset_loaded(df):
    assert df.shape[0] == 569
    assert "target" in df.columns
    assert df.isnull().sum().sum() == 0


def test_at_least_five_features_selected(df):
    features = tm.select_features(df)
    assert len(features) >= 5
    assert "target" not in features


def test_six_models_defined():
    assert len(tm.get_model_configs()) == 6


def test_models_reach_reasonable_accuracy(df):
    data = tm.split_and_scale(df, tm.select_features(df))
    for name, (model, scaled, _) in tm.get_model_configs().items():
        X_train, X_test, y_train, y_test, X_train_s, X_test_s = data
        Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
        model.fit(Xtr, y_train)
        metrics = tm.evaluate(model, Xte, y_test)
        assert metrics["accuracy"] > 0.85, f"{name} accuracy too low: {metrics['accuracy']:.3f}"


def test_full_pipeline_registers_two_versions(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = f"file:{tmp_path / 'mlruns'}"
    summary = tm.run_pipeline(tracking_uri=uri, results_path=str(tmp_path / "results.json"))
    assert len(summary["all_models"]) == 6
    versions = MlflowClient(tracking_uri=uri).search_model_versions(f"name='{tm.REGISTERED_MODEL_NAME}'")
    assert sorted(int(v.version) for v in versions) == [1, 2]
