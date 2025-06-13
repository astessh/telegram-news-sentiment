import subprocess
from pathlib import Path

import gdown

GDRIVE_FILE_ID = "12VUs0oCDo4ODmGa3q8BMNDT_p76LEwQu"
JSON_PATH = Path("data/raw/data.json")


def download():
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(JSON_PATH), quiet=False)


def add_to_dvc():
    subprocess.run(["dvc", "add", str(JSON_PATH)], check=True)


def load_data():
    download()
    add_to_dvc()


if __name__ == "__main__":
    load_data()
