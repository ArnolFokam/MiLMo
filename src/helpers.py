import os
import random
import string


def get_dir(*paths) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    Returns:
        str:
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory


def generate_random_string(length: int = 5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
