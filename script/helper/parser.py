import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, help='Name of dataset (torgo, hu, mls_de, mls_en)')
    parser.add_argument('-l', type=str, help='Language (de, en)')
    parser.add_argument('-m', type=str, help='Model')
    parser.add_argument('-s', type=str, help='Names of speakers for fine-tuning in \" \"')
    parser.add_argument('-local', type=bool, default=False, help='True if local')
    parser.add_argument('-llo', type=bool, default=False, help='True if only last two layers should be changed')
    parser.add_argument('-sd', type=str, default='', help='Name of second dataset (torgo, hu, mls_de, mls_en)')
    parser.add_argument('-optuna', type=bool, help='True if optuna is to be used', default=False)

    parser.add_argument('-bs', type=int, default=8)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-epoch', type=int, default=30)

    return parser.parse_args()
