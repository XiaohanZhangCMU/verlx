import os
#from composer.utils import dist
import torch.distributed as dist
import torch
import time
import subprocess
import ray
import datetime
import socket

def get_ip():
    """Retrieve the IP address of the current node."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception:
        return "127.0.0.1"

def get_local_rank():
    """Retrieve the local rank of the process."""
    return int(os.environ.get("LOCAL_RANK", 0))

def get_global_rank():
    """Retrieve the global rank of the process."""
    return dist.get_rank() if dist.is_initialized() else 0

def initialize_ray_cluster():
    # Ensure NCCL doesn't block
    #os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    #dist.initialize_dist()

    #os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    #os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=20))

    torch.cuda.set_device(f'cuda:{get_local_rank()}')

    command = "ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    #result.stdout.strip()

    ip_address = get_ip()

    #head_ip_address = dist.all_gather_object(ip_address)[0]
    # Gather IP addresses from all nodes
    gathered_ips = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_ips, ip_address)
    head_ip_address = gathered_ips[0]  # Use rank 0 as head

    print(f"bigning debug {gathered_ips=}, {ip_address=}, {result.stdout=}, {get_global_rank()=}")

    dist.barrier()

    if get_local_rank() == 0 and get_global_rank() == 0:
        subprocess.run('ray start --head', shell=True)
        ray.init()

    dist.barrier()

    if get_local_rank() == 0 and get_global_rank() != 0:
        time.sleep(10)
        print(f"bigning debug {head_ip_address=}, {get_global_rank()=}")
        subprocess.run(f'ray start --address {head_ip_address}:6379', shell=True)
        print(f"bigning debug ray start done")
        ray.init(address=f'{head_ip_address}:6379')

    dist.barrier()


if __name__ == '__main__':
    initialize_ray_cluster()

    print('Ray started. Now running code.')

    if get_global_rank() == 0:
        @ray.remote
        def test_task(x):
            return f"Ray worker {ray.get_runtime_context().node_id} processed value: {x}"

        # Run a simple test task on Ray
        futures = [test_task.remote(i) for i in range(20)]
        results = ray.get(futures)

        print("Ray Test Results:")
        for res in results:
            print(res)

        print(f"Rank {get_global_rank()} shutting down Ray...")
        ray.shutdown()

    dist.barrier()  # Ensure all processes complete Ray execution before teardown
    # Destroy NCCL process group safely
    print(f"Rank {get_global_rank()} destroying NCCL process group...")
    dist.distributed.destroy_process_group()

    print(f"Rank {get_global_rank()} successfully cleaned up.")

