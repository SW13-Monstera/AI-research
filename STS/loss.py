# from torch import backends
# from torch import corrcoef
# def pearson_loss(y_true, y_pred):

from sklearn.metrics import mean_squared_error


def compute_metric(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, squared=False)
    return {"mse": mse}
