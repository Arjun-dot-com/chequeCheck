from .banking_records import load_banking_records


def retrieve_account(account_number):
    records = load_banking_records()

    for record in records:
        if record["account_number"] == account_number:
            return record

    return None


def retrieve_cheque(account_number, cheque_number):
    account = retrieve_account(account_number)

    if account is None:
        return None

    for cheque in account.get("cheques", []):
        if cheque["cheque_number"] == cheque_number:
            return {
                "account": account,
                "cheque": cheque
            }

    return None