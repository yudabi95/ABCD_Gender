import torch
from utils import evaluate_test_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


evaluate_test_accuracy()