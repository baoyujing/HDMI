import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedder', nargs='?', default='HDMI', help='HDMI or HDI')
    parser.add_argument('--hid_units', type=int, default=128, help='hidden dimension')
    parser.add_argument('--same_discriminator', type=bool, default=False,
                        help='whether to use the same discriminator for the layers and fusion module')

    parser.add_argument('--dataset', nargs='?', default='amazon')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')

    parser.add_argument('--nb_epochs', type=int, default=10000, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--coef_layers', type=list, default=[1, 2, 0.001],
                        help='different layers of the multiplex network: '
                             'coefficients for the extrinsic, intrinsic and joint signals')
    parser.add_argument('--coef_fusion', type=list, default=[0.01, 0.1, 0.001],
                        help='fusion module: coefficient for the extrinsic, intrinsic and joint signals')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')

    return parser.parse_known_args()


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)


def main():
    args, unknown = parse_args()
    printConfig(args)

    if args.embedder == "HDI":
        from models import HDI
        embedder = HDI(args)
    elif args.embedder == "HDMI":
        from models import HDMI
        embedder = HDMI(args)

    macro_f1s, micro_f1s, k1 = embedder.training()
    return macro_f1s, micro_f1s, k1


if __name__ == '__main__':
    macro_f1s, micro_f1s, k1 = main()
