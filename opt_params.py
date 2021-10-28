import argparse

parser = argparse.ArgumentParser(description='SLvsML')
parser.add_argument('--tr_smp_sizes', nargs="*", type=int,
                    default=(100, 200, 500, 1000, 2000, 5000, 10000), help='')
parser.add_argument('--nReps', type=int, default=20, metavar='N',
                    help='random seed (default: 20)')
parser.add_argument('--nc', type=int, default=2, metavar='N',
                    help='number of classes in dataset (default: 10)')
parser.add_argument('--bs', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--es', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--pp', type=int, default=0, metavar='N',
                    help='iteration flag to allow multiple  processes')
parser.add_argument('--es_va', type=int, default=1, metavar='N',
                    help='1: use val accuracy; 0: use val loss : default on i.e. uses val accuracy')
parser.add_argument('--es_pat', type=int, default=50, metavar='N',
                    help='patience for early stopping (default: 20)')
parser.add_argument('--ml', default='./temp/', metavar='S',
                    help='model location (default: ./temp/)')
parser.add_argument('--mt', default='AlexNet3D', metavar='S',
                    help='modeltype (default: AlexNet3D)')
parser.add_argument('--ssd', default='/SampleSplits/', metavar='S',
                    help='sample size directory (default: /SampleSplits/)')
parser.add_argument('--scorename', default='sex_at_birth', metavar='S',
                    help='scorename (default: fluid_intelligence_score_f20016_2_0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='turn off to enable CUDA training')
parser.add_argument('--nw', type=int, default=8, metavar='N',
                    help='number of workers (default: 8)')
parser.add_argument('--cr', default='clx', metavar='S',
                    help='classification (clx) or regression (reg) - (default: clx)')
parser.add_argument('--tss', type=int, default=100, metavar='N',
                    help='training sample size (default: 100)')
parser.add_argument('--rep', type=int, default=0, metavar='N',
                    help='crossvalidation rep# (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')

args = parser.parse_args()