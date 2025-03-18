from verl.utils.tracking import Tracking

import multiprocessing

import atexit
import torch

def cleanup():
    print("[Cleanup] Ensuring all processes close properly...")
    for p in multiprocessing.active_children():
        p.terminate()
        p.join()

atexit.register(cleanup)
method = multiprocessing.get_start_method(allow_none=True)
if method != "spawn":
    print(f'start method for multiprocessing: {method}')
    #multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__":
    tracking = Tracking(project_name="test_project", experiment_name="test_experiment", default_backend="mlflow")
    tracking.log({"loss": 0.123, "accuracy": 0.98}, step=1)
    print("MLflow logging test complete!")

#    import mlflow
#    mlflow.flush_async_logging()


