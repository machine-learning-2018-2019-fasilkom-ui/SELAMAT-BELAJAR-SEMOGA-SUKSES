import multiprocess as mp
import sys

def sequential_execute(output_queue, func, list_of_kwargs):
    # for kwargs in list_of_kwargs:
    #     output_queue.put(func(**kwargs))
    process = mp.current_process()
    print('sequential execute: executing on PID', process.pid)
    sys.stdout.flush()
    result = [func(**kwargs) for kwargs in list_of_kwargs]
    # result = []
    # return result
    output_queue.put(result)
