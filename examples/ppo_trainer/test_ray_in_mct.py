import os
from composer.utils import dist
import torch
import time
import subprocess
import ray

def initialize_ray_cluster():
    # Ensure NCCL doesn't block
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    dist.initialize_dist()

    command = "ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    ip_address = result.stdout.strip()

    head_ip_address = dist.all_gather_object(ip_address)[0]

    dist.barrier()

    if dist.get_local_rank() == 0 and dist.get_global_rank() == 0:
        subprocess.run('ray start --head', shell=True)
        ray.init()

    dist.barrier()

    if dist.get_local_rank() == 0 and dist.get_global_rank() != 0:
        time.sleep(10)
        subprocess.run(f'ray start --address {head_ip_address}:6379', shell=True)
        ray.init(address=f'{head_ip_address}:6379')

    dist.barrier()


if __name__ == '__main__':
    initialize_ray_cluster()

    print('Ray started. Now running code.')

    if dist.get_global_rank() == 0:
        @ray.remote
        def test_task(x):
            return f"Ray worker {ray.get_runtime_context().node_id} processed value: {x}"

        # Run a simple test task on Ray
        futures = [test_task.remote(i) for i in range(5)]
        results = ray.get(futures)

        print("Ray Test Results:")
        for res in results:
            print(res)

    dist.barrier()  # Ensure all processes complete Ray execution before teardown

    # Properly shut down Ray before NCCL
    print(f"Rank {dist.get_global_rank()} shutting down Ray...")
    ray.shutdown()

    dist.barrier()  # Ensure Ray has shut down on all processes

    # Destroy NCCL process group safely
    print(f"Rank {dist.get_global_rank()} destroying NCCL process group...")
    torch.distributed.destroy_process_group()

    print(f"Rank {dist.get_global_rank()} successfully cleaned up.")

