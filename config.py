from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='region gan')

    parser.add_argument('--path',
                        type=str,
                        default='../lmdbs/art_landscape_1k',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda',
                        type=int,
                        default=0,
                        help='index of gpu to use')
    parser.add_argument('--name',
                        type=str,
                        default='test1',
                        help='experiment name')
    parser.add_argument('--iter',
                        type=int,
                        default=50000,
                        help='number of iterations')
    parser.add_argument('--start_iter',
                        type=int,
                        default=0,
                        help='the iteration to start training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='mini batch number of images')
    parser.add_argument('--im_size',
                        type=int,
                        default=1024,
                        help='image resolution')
    parser.add_argument('--ckpt',
                        type=str,
                        default='None',
                        help='checkpoint weights path if have one')
    parser.add_argument('--ndf',
                        type=int,
                        default=64,
                        help='')
    parser.add_argument('--ngf',
                        type=int,
                        default=64,
                        help='')
    parser.add_argument('--nlr',
                        type=int,
                        default=0.0002,
                        help='')
    parser.add_argument('--nz',
                        type=int,
                        default=256,
                        help='Latent dimension')
    parser.add_argument('--nbeta1',
                        type=float,
                        default=0.5,
                        help='')
    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--save_interval',
                        type=int,
                        default=100,
                        help='')

    args = parser.parse_args()

    return args
