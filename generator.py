import argparse
import os
from hash_function.argon2_utils import generate_hash_argon2
from hash_function.sha256_utils import generate_hash_sha256
from utilities import clear_output_files

def main():
    parser = argparse.ArgumentParser(description='Generate password hashes.')
    parser.add_argument('--argon2', action='store_true', help='Use Argon2 algorithm for hashing')
    parser.add_argument('--sha256', action='store_true', help='Use SHA256 algorithm for hashing')
    args = parser.parse_args()

    clear_output_files('data/unsalted_hashes.txt', 'data/salted_hashes.txt')

    with open('data/passwords_config.txt', 'r') as config_file:
        password_files = [line.strip() for line in config_file.readlines()]

    if args.argon2:
        hash_function = generate_hash_argon2
    elif args.sha256:
        hash_function = generate_hash_sha256
    else:
        raise ValueError("No hashing algorithm specified.")

    for password_file in password_files:
        with open(password_file, 'r') as file:
            passwords = [line.strip() for line in file.readlines()]

        with open('data/unsalted_hashes.txt', 'a') as file:
            for password in passwords:
                file.write(hash_function(password) + '\n')

        with open('data/salted_hashes.txt', 'a') as file:
            for password in passwords:
                salt = os.urandom(16).hex()
                file.write(hash_function(password, salt) + '\n')

if __name__ == "__main__":
    main()
