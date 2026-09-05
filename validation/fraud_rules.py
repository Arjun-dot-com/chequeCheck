def apply_fraud_rules(
    extracted_data,
    cv_validation,
    account,
    cheque,
    mismatches
):
    risk_score = 0
    reasons = []

    # 1. Account status
    if account.get("status") != "active":
        risk_score += 50
        reasons.append("Account is not active")

    # 2. Duplicate cheque
    if cheque.get("status") == "used":
        risk_score += 70
        reasons.append("Cheque has already been processed")

    # 3. Cheque number mismatch
    if "Cheque number mismatch" in mismatches:
        risk_score += 50
        reasons.append("Cheque number does not match banking record")

    # 4. Payee mismatch
    if "Payee mismatch" in mismatches:
        risk_score += 30
        reasons.append("Payee does not match banking record")

    # 5. Amount mismatch
    if "Amount mismatch" in mismatches:
        risk_score += 40
        reasons.append("Amount does not match banking record")

    # 6. Date mismatch
    if "Date mismatch" in mismatches:
        risk_score += 20
        reasons.append("Date does not match banking record")

    # 7. Invalid OCR amount
    if "Invalid amount extracted by OCR" in mismatches:
        risk_score += 25
        reasons.append("Amount could not be reliably extracted")

    # 8. Signature check
    if not cv_validation.get("is_signed", False):
        risk_score += 40
        reasons.append("Signature is missing")

    # 9. Amount tampering
    if cv_validation.get("amount_tamper_flag", False):
        risk_score += 50
        reasons.append("Possible amount tampering detected")

    # 10. Payee tampering
    if cv_validation.get("payee_tamper_flag", False):
        risk_score += 40
        reasons.append("Possible payee tampering detected")

    return risk_score, reasons