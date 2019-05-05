from keras import backend as K


def precision(y_true, y_pred):


    #predicted positives
    predictions = K.round(y_pred)
    predicted_positives = K.sum(predictions)

    #true positives
    true_positives = K.sum(K.round(y_true * predictions))
    P = true_positives / (predicted_positives + K.epsilon())
    return P

def recall(y_true, y_pred):


    #predicted positives
    predictions = K.round(y_pred)

    #all positives
    all_positives = K.sum(y_true)

    #true positives
    true_positives = K.sum(K.round(y_true * predictions))

    R = true_positives / all_positives
    return R

def f1(y_true, y_pred):


    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    return 2*((P*R)/(P+R+K.epsilon()))


# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.
#
#         Only computes a batch-wise average of recall.
#
#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         print("y_true ", y_true.eval())
#         print("y_pred ", y_pred.eval())
#
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#
#     def precision(y_true, y_pred):
#         """Precision metric.
#
#         Only computes a batch-wise average of precision.
#
#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
#
#
# def recall(y_true, y_pred):
#     """Recall metric.
#
#     Only computes a batch-wise average of recall.
#
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     print("y_true ", y_true)
#     print("y_pred ", y_pred)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision(y_true, y_pred):
#     """Precision metric.
#
#     Only computes a batch-wise average of precision.
#
#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def tprate(y_true, y_pred):
#     return recall(y_true, y_pred)
#
# def fprate2(y_true, y_pred):
#     #invert true and negative so negative is 1  and true is 1.
#     y_true = 1 - y_true
#     #invert predictions so that we get 1 for the predictions originally set to 0
#     y_pred = 1 - y_pred
#     return recall(y_true, y_pred)

def fprate(y_true, y_pred):
    #predicted positives
    predictions = K.round(y_pred)
    predicted_positives = K.sum(predictions)

    #all positives
    all_positives = K.sum(y_true)

    #true positives
    true_positives = K.sum(K.round(y_true * predictions))

    false_positive = predicted_positives - true_positives

    #negatives
    y_false = 1 - y_true

    all_negatives = K.sum(y_false)
    fpr = false_positive / (all_negatives + K.epsilon())
    return fpr


# def fprate(y_true, y_pred):
#     # true positives
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     # predicted_positives = true_positives + false_positives
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     # false_positives
#     false_positive = predicted_positives - true_positives
#     # Now work on negatives
#     y_false = 1 - y_true
#     possible_negatives = K.sum(K.round(K.clip(y_false, 0, 1)))
#     fprate = false_positive / (possible_negatives  + K.epsilon())
#     return fprate

def accuracy(y_true, y_pred):
    n_samples = len(y_pred)
    correct = 1 * (y_true == y_pred)
    return sum(correct) / n_samples


def calculate_cm_metrics(c_matrix, track):
    prec_0 = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[1][0] + c_matrix[2][0])
    prec_1 = c_matrix[1][1] / (c_matrix[1][1] + c_matrix[0][1] + c_matrix[2][1])
    prec_2 = c_matrix[2][2] / (c_matrix[2][2] + c_matrix[1][2] + c_matrix[0][2])

    recall_0 = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1] + c_matrix[0][2])
    recall_1 = c_matrix[1][1] / (c_matrix[1][1] + c_matrix[1][0] + c_matrix[1][2])
    recall_2 = c_matrix[2][2] / (c_matrix[2][2] + c_matrix[2][0] + c_matrix[2][1])

    f1_0 = 2 * ((prec_0 * recall_0) / (prec_0 + recall_0))
    f1_1 = 2 * ((prec_1 * recall_1) / (prec_1 + recall_1))
    f1_2 = 2 * ((prec_2 * recall_2) / (prec_2 + recall_2))

    tn_0 = c_matrix[1][1] + c_matrix[1][2] + c_matrix[2][1] + c_matrix[2][2]
    tn_1 = c_matrix[0][0] + c_matrix[0][2] + c_matrix[2][0] + c_matrix[2][2]
    tn_2 = c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1]

    spec_0 = tn_0 / (tn_0 + c_matrix[1][0] + c_matrix[2][0])
    spec_1 = tn_1 / (tn_1 + c_matrix[0][1] + c_matrix[2][1])
    spec_2 = tn_2 / (tn_2 + c_matrix[0][2] + c_matrix[1][2])

    t = track + ("Precision 0: {}\n" 
                "Precision 1: {}\n"
                "Precision 2: {}\n"
                "Recall 0: {}\n"
                "Recall 1: {}\n"
                "Recall 2: {}\n"
                "F1 Score 0: {}\n"
                "F1 Score 1: {}\n"
                "F1 Score 2: {}\n"
                "Specificity 0: {}\n"
                "Specificity 1: {}\n"
                "Specificity 2: {}\n").format(prec_0, prec_1, prec_2, recall_0, recall_1, recall_2, f1_0, f1_1,
                                                f1_2, spec_0, spec_1, spec_2)
    return prec_1, recall_1, f1_1, spec_1, t
