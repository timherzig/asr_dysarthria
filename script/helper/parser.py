import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, help='Name of dataset (torgo, hu, mls_de, mls_en)')
    parser.add_argument('-l', type=str, help='Language (de, en)')
    parser.add_argument('-m', type=str, help='Model')
    parser.add_argument('-local', type=bool, default=False, help='True if local')

    return parser.parse_args()
