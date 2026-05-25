import torch.nn.functional as F


def cosine_distance(a, b):
    return (1 - F.cosine_similarity(a, b, dim=-1)).mean()


def summary_loss(student, teacher):
    return cosine_distance(student, teacher)


def feature_loss(student, teacher):
    return cosine_distance(student, teacher) + F.smooth_l1_loss(student, teacher)
