import torch.cuda
from utils import *
from models.ResNet_Model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device('cuda:1')
PATH = '/home/users/ybi3/SMLvsDL/Pretrained_sgd5.pt'


def main():


    model = loadNet()
    model.to(device)

    # Loss and optimizer

    generate_validation(model)

    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    main()



