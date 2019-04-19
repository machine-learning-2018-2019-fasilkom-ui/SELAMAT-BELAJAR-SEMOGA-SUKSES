import multiprocess as mp
import sys

def sequential_execute(output_queue, func, list_of_kwargs):
    # process = mp.current_process()
    # print('sequential execute: executing on PID', process.pid)
    # sys.stdout.flush()
    for kwargs in list_of_kwargs:
        output_queue.put(func(**kwargs))
