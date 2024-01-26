import hashlib
import time

def generate_hash_sha256(password, salt=None):
    if salt:
        password = password + salt
    return hashlib.sha256(password.encode()).hexdigest()

def generate_sha256_rainbow_table(passwords):
    rainbow_table = {}
    total_time_cost = 0
    for password in passwords:
        start_time = time.time()
        hash_value = hashlib.sha256(password.encode()).hexdigest()
        end_time = time.time()

        hashing_time = end_time - start_time
        # print(f"SHA256 Hashing Time: {hashing_time:.4f} seconds")
        total_time_cost += hashing_time
        rainbow_table[hash_value] = password
    return rainbow_table, total_time_cost