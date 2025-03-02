from verl.utils.tracking import Tracking

import multiprocessing

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    print('change to spawn')
    multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__":
    tracking = Tracking(project_name="test_project", experiment_name="test_experiment", default_backend="mlflow")
    tracking.log({"loss": 0.123, "accuracy": 0.98}, step=1)
    print("MLflow logging test complete!")

    import mlflow
    mlflow.flush_async_logging()


