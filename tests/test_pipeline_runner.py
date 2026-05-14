from app.pipeline.pipeline_runner import PipelineRunner

def test_pipeline_runner():
    runner = PipelineRunner(
        file_path="data/bhubaneswar_rent_clean_audited.csv",
        target_column="rent",
        task_type="regression"
    )

    output = runner.run()

    assert output["status"] == "success"
    assert "metadata" in output
    assert "checks" in output
    assert "recommendations" in output
    assert "model_metrics" in output
    assert "feature_importance" in output
    
    # Verify that K-Fold Cross Validation ran successfully
    assert "cv_mean_rmse" in output["model_metrics"]
    assert "cv_std_rmse" in output["model_metrics"]