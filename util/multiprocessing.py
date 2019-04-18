
def sequential_execute(func, list_of_kwargs):
    # for kwargs in list_of_kwargs:
    #     output_queue.put(func(**kwargs))
    result = [func(**kwargs) for kwargs in list_of_kwargs]
    return result
