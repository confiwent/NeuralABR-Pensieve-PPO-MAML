import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=True)
parser.add_argument('--n', nargs='?', default=1, const=0)
args = parser.parse_args()

if args.debug:
    print('Debug mode on')
    print(args.n)
