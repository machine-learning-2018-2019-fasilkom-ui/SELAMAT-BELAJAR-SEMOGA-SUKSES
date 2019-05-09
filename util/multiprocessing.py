
def sequential_execute(output_queue, func, list_of_kwargs):
    for kwargs in list_of_kwargs:
        output_queue.put(func(**kwargs))
