from app.pipeline.pipeline_runner import PipelineRunner


def test_pipeline_runner():
    runner = PipelineRunner(
        file_path="data/bhubaneswar_rent_clean_audited.csv",
        target_column="area",
        task_type="classification"
    )

    output = runner.run()

    assert "metadata" in output
    assert "checks" in output
    assert "recommendations" in output
    assert "model_metrics" in output
    assert "feature_importance" in output