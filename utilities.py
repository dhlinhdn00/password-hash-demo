
def load_hashes(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def load_rainbow_table(file_path):
    rainbow_table = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue

            try:
                hash, password = line.split()
                rainbow_table[hash] = password
            except ValueError:
                continue
    return rainbow_table

def write_rainbow_table(rainbow_table, output_file):
    """
    Write the rainbow table to a file.

    :param rainbow_table: The rainbow table to write.
    :param output_file: The file to write the rainbow table to.
    """
    with open(output_file, 'a') as file:
        for password, hash_value in rainbow_table.items():
            file.write(f"{password} {hash_value}\n")

def clear_output_files(*files):
    """
    Clears the contents of the specified files.

    :param files: A list of file paths to clear.
    """
    for file in files:
        open(file, 'w').close()
