import argparse
from utilities import load_hashes, load_rainbow_table


def crack_hash(hash, rainbow_table):
    return rainbow_table.get(hash)

def main():
    parser = argparse.ArgumentParser(description='Crack salted and unsalted hashes.')
    parser.add_argument('--salted', action='store_true', help='Choose to crack salted hashes')
    parser.add_argument('--unsalted', action='store_true', help='Choose to crack unsalted hashes')
    args = parser.parse_args()

    if args.salted:
        hash_file = 'data/salted_hashes.txt'
    elif args.unsalted:
        hash_file = 'data/unsalted_hashes.txt'

    hashes = load_hashes(hash_file)
    rainbow_table = load_rainbow_table('data/rainbow_table.txt')

    for hash in hashes:
        password = crack_hash(hash, rainbow_table)
        if password:
            print(f"Hash: {hash} -> Password: {password}")
        else:
            print(f"Hash: {hash} -> Uncrackable.")

if __name__ == "__main__":
    main()
