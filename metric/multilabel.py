# multilabel accuracy
# https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data


def multilabel_accuracy(Y_true, Y_pred):
    assert(len(Y_true) == len(Y_pred))

    n = len(Y_true)
    result = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        true_set = set(y_true)
        pred_set = set(y_pred)

        intersection = len(true_set.intersection(pred_set))
        union = len(true_set.union(pred_set))
        result += intersection/union

    return result/n