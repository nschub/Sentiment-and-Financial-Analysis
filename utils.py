from datetime import datetime
import os


def save_response_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as file:
        file.write(data + '\n')

def get_api_key(filename='API-key.txt'):
    with open(filename, 'r') as file:
        return file.read().strip()

def generate_filename(basename, extension, prefix="Output_Data/Logfiles", timestamp=None):
    """Generates a filename with a timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{basename}_{timestamp}.{extension}"
    return os.path.join(prefix, filename)


