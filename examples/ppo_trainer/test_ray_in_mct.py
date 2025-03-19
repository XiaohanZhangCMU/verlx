import os
from composer.utils import dist
import torch
import time
import subprocess
import ray

def initialize_ray_cluster():
    dist.initialize_dist(backend="gloo")

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

    print('ray started. now running code')

    if dist.get_global_rank() == 0:
        @ray.remote
        def test_task(x):
            return f"Ray worker {ray.get_runtime_context().node_id} processed value: {x}"

        # Initialize Ray
        ray.init(address="auto")

        # Run a simple test task on Ray
        futures = [test_task.remote(i) for i in range(5)]
        results = ray.get(futures)

        print("Ray Test Results:")
        for res in results:
            print(res)

        # Shutdown Ray to clean up resources
        ray.shutdown()

    dist.barrier()
    torch.distributed.destroy_process_group()

