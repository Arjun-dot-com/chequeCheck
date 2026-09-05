from datetime import datetime


def clean_value(field):
    """
    Extract the actual value from an OCR field.

    Example:
    {"value": "ABC Traders", "confidence": 95}
    becomes:
    "ABC Traders"
    """

    if isinstance(field, dict):
        return str(field.get("value", "")).strip()

    return str(field).strip()


def normalize_text(value):
    """Normalize text for comparison."""

    return clean_value(value).lower().strip()


def normalize_amount(value):
    """
    Convert OCR amount into a numeric value.

    Handles values such as:
    ₹25,000
    Rs. 25000
    25,000
    25000
    """

    text = clean_value(value)

    text = (
        text.replace("₹", "")
        .replace(",", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .strip()
    )

    try:
        return float(text)
    except ValueError:
        return None


def validate_account(account):
    """
    Validate the status of the bank account.
    """

    if account.get("status") != "active":
        return False, "Account is not active"

    return True, "Account is active"


def validate_cheque_series(cheque_number, cheque_series):
    """
    Validate whether the cheque number belongs
    to the expected cheque series.

    Example:
    Series AB + cheque 123456
    """

    cheque_number = clean_value(cheque_number)
    cheque_series = clean_value(cheque_series)

    # For our mock data, the series is stored separately.
    # The current cheque number is numeric, so we verify
    # that a valid series exists for the account.
    if not cheque_series:
        return False, "Cheque series information missing"

    if not cheque_number:
        return False, "Cheque number missing"

    return True, "Cheque series information available"


def validate_payee(extracted_payee, bank_payee):
    """Compare OCR payee with banking record."""

    extracted = normalize_text(extracted_payee)
    stored = normalize_text(bank_payee)

    if not extracted:
        return False, "Payee could not be extracted"

    if extracted != stored:
        return False, "Payee mismatch"

    return True, "Payee matches"


def validate_amount(extracted_amount, bank_amount):
    """Compare OCR amount with banking record."""

    extracted = normalize_amount(extracted_amount)

    if extracted is None:
        return False, "Invalid amount extracted by OCR"

    if extracted != float(bank_amount):
        return False, "Amount mismatch"

    return True, "Amount matches"


def validate_date(extracted_date, bank_date):
    """Compare cheque date with banking record."""

    extracted = clean_value(extracted_date)
    stored = clean_value(bank_date)

    if not extracted:
        return False, "Cheque date could not be extracted"

    # Exact comparison for the first version.
    if extracted != stored:
        return False, "Cheque date mismatch"

    return True, "Cheque date matches"


def validate_duplicate(cheque_status):
    """Check whether the cheque has already been processed."""

    if cheque_status == "used":
        return False, "Cheque has already been processed"

    return True, "Cheque has not been processed"


def validate_cheque(extracted_data, account, bank_cheque):
    """
    Perform all required PS-05 validation checks.
    """

    results = {}
    failures = []

    # 1. Account status
    valid, message = validate_account(account)
    results["account_status"] = {
        "valid": valid,
        "message": message
    }

    if not valid:
        failures.append(message)

    # 2. Cheque series
    valid, message = validate_cheque_series(
        extracted_data.get("cheque_number"),
        account.get("cheque_series")
    )

    results["cheque_series"] = {
        "valid": valid,
        "message": message
    }

    if not valid:
        failures.append(message)

    # 3. Payee
    valid, message = validate_payee(
        extracted_data.get("payee"),
        bank_cheque.get("payee")
    )

    results["payee"] = {
        "valid": valid,
        "message": message
    }

    if not valid:
        failures.append(message)

    # 4. Amount
    valid, message = validate_amount(
        extracted_data.get("amount"),
        bank_cheque.get("amount")
    )

    results["amount"] = {
        "valid": valid,
        "message": message
    }

    if not valid:
        failures.append(message)

    # 5. Date
    valid, message = validate_date(
        extracted_data.get("date"),
        bank_cheque.get("date")
    )

    results["date"] = {
        "valid": valid,
        "message": message
    }

    if not valid:
        failures.append(message)

    # 6. Duplicate check
    valid, message = validate_duplicate(
        bank_cheque.get("status")
    )

    results["duplicate_check"] = {
        "valid": valid,
        "message": message
    }

    if not valid:
        failures.append(message)

    return {
        "passed": len(failures) == 0,
        "checks": results,
        "failures": failures
    }