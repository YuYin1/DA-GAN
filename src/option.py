import argparse

parser = argparse.ArgumentParser(description='DA-GAN')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../Dataset', help='dataset directory')
parser.add_argument('--data_train', type=str, default='MultiPIE', help='train dataset name')
parser.add_argument('--data_test', type=str, default='MultiPIE_15+MultiPIE_30+MultiPIE_45+MultiPIE_60', help='test dataset name')
parser.add_argument('--patch_size', type=int, default=128, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')

# Model specifications
parser.add_argument('--model', default='MFSR', help='model name')
parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.', help='pre-trained model directory')
parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')

# Training specifications
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1, help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1.0e-4, help='learning rate')
parser.add_argument('--decay', type=str, default='8-16', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')

# Loss specifications 
parser.add_argument('--loss', type=str, default='10*L1+0.1*WGAN_GP+0.1*GAN_FACE+0.1*GAN_HAIR+0.1*GAN_ELE+0.1*IP+0.001*VGG54+0.0001*TV', help='loss function configuration') #SoftArgmax
parser.add_argument('--skip_threshold', type=float, default='1e8', help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='SI_dual_attention', help='file name to save')
parser.add_argument('--load', type=str, default='', help='file name to load')
parser.add_argument('--resume', type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=1000, help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', help='save output results')
parser.add_argument('--save_gt', action='store_true', help='save low-resolution and high-resolution images together')

args = parser.parse_args()

args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

