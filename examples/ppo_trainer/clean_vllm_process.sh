set -x
ray stop --force
top -b -n 1 | grep pt_main_thread | awk '{print $1}' | xargs kill -9
