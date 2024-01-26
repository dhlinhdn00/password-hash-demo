from argon2 import PasswordHasher
import hashlib
import time

def extract_hash_from_argon2(full_hash):
    parts = full_hash.split('$')
    if len(parts) < 6 or not full_hash.startswith('$argon2'):
        raise ValueError("Not a valid Argon2 string!")
    return parts[-1]

def generate_hash_argon2(password, salt=None):
    hasher = PasswordHasher()
    if salt:
        full_hash = hasher.hash(password + salt)
    else:
        full_hash = hasher.hash(password)
    return extract_hash_from_argon2(full_hash)

def generate_argon2_rainbow_table(passwords):
    hasher = PasswordHasher()
    rainbow_table = {}
    total_time_cost = 0
    for password in passwords:
        start_time = time.time()
        full_hash = hasher.hash(password)
        hash_part = extract_hash_from_argon2(full_hash)
        end_time = time.time()

        hashing_time = end_time - start_time
        # print(f"Argon2 Hashing Time: {hashing_time:.4f} seconds")
        rainbow_table[hash_part] = password
        total_time_cost += hashing_time
    return rainbow_table, total_time_cost