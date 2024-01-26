import argparse
import os
from hash_function.argon2_utils import generate_argon2_rainbow_table
from hash_function.sha256_utils import generate_sha256_rainbow_table
from utilities import write_rainbow_table

def main():
    total_time = 0
    parser = argparse.ArgumentParser(description="Generate a rainbow table.")
    parser.add_argument("--sha256", action="store_true", help="Use SHA-256 for hashing")
    parser.add_argument("--argon2", action="store_true", help="Use Argon2 for hashing")
    args = parser.parse_args()

    with open('data/passwords_config.txt', 'r') as config_file:
        password_files = [line.strip() for line in config_file.readlines()]

    open('data/rainbow_table.txt', 'w').close()

    for password_file in password_files:
        with open(password_file, 'r') as file:
            passwords = [line.strip() for line in file.readlines()]

        if args.sha256:
            rainbow_table, time_cost = generate_sha256_rainbow_table(passwords)
        elif args.argon2:
            rainbow_table, time_cost = generate_argon2_rainbow_table(passwords)
        else:
            raise ValueError("No hashing algorithm specified")
        
        total_time += time_cost
        write_rainbow_table(rainbow_table, 'data/rainbow_table.txt')
    print(f" Total time = {total_time} s")


if __name__ == "__main__":
    main()
