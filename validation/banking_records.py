import json
from pathlib import Path


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "bank_data"
    / "banking_records.json"
)


def load_banking_records():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)