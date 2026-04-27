from src.data_ingestion import load_params


def test_load_params_reads_top_level_mapping():
    params = load_params("params.yaml")
    assert "data" in params