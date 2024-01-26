import os

def update_passwords_config(directory, config_file):
    """
    Update the passwords_config.txt file with the paths of files in the given directory.

    :param directory: The directory to scan for files.
    :param config_file: The configuration file to update.
    """
    files = os.listdir(directory)

    with open(config_file, 'w') as file:
        for filename in files:
            file_path = os.path.join(directory, filename)
            file.write(file_path + '\n')
    print("Update passwords config completely!")

if __name__ == "__main__":
    update_passwords_config('data/passwords', 'data/passwords_config.txt')
