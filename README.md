# Passhashdemo
Password Hash Demo - A02: 2021 - Cryptographic Failures-OWASP - Information Security Course

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PASSWORD-HASH-DEMO
We have 2 main parts:
1. Hash and attack using rainbow_table.
2. Hash and attack using brute force on GPU support device.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

You can see in folder data and modify only passwords_config.txt.
Assume that we have a password path per line.

Install libs for environments:
$ pip install -r requirements

And update the passwords config by:
$ python update_config.py
So you must update the passwirds config after adding or changing something in data/passwords for demo.

1. Run the generator.py file as: 
$ python generator.py --{hash function you want to implement}

In this project, I only implement 2 hash functions: argon2 and sha256. So in future, I will add some hash functions for comparisons.
Then, create rainbow table by:
$ python rainbow_table.py --{hash function you want to implement}

Finally, crack them:
$ python cracker.py --{salted/unsalted -- noted: choose only one!}

2. In additon, I implement how to hash sha256 by bruteforce using CUDA. I built and run it like this:
$ nvcc bruteforce/sha256_main.cu -o sha256

$ ./sha256 ./data/passwords_config.txt

But, in constraint, We cannot yet build a function that uses the gpu to perform brute force on the sha256 hash code (it's so complicated!). Instead of that, I clone the git hub source for brute-force attack task: https://github.com/ngirot/BruteForce

After cloning it in folder bruteforce, the command for running it:
$ bruteforce/BruteForce/BruteForce --type sha256 --value {hash value you wanna to attack} --gpu

hash value: you can get it in data/unsalted_hashes.txt.

My implementation is conducted on a personal computer: Core I5 11th, 16GB RAM and NVIDIA GeForce RTX 3050 4g VRAM.

