from .retriever import retrieve_cheque
from .validator import validate_cheque
from .fraud_rules import apply_fraud_rules
from .decision_engine import make_decision


def process_validation(scan_result):
    extracted_data = scan_result.get("extracted_data", {})
    cv_validation = scan_result.get("validation", {})

    cheque_number = (
        extracted_data
        .get("cheque_number", {})
        .get("value", "")
        .strip()
    )

    account_number = (
        extracted_data
        .get("micr_code", {})
        .get("value", "")
        .strip()
    )

    # Check required identifiers
    if not cheque_number:
        return {
            "decision": "REJECT",
            "risk_score": 100,
            "matched": False,
            "reasons": [
                "Cheque number could not be extracted"
            ]
        }

    if not account_number:
        return {
            "decision": "REJECT",
            "risk_score": 100,
            "matched": False,
            "reasons": [
                "Account identifier could not be extracted"
            ]
        }

    # Retrieve banking record
    matched_record = retrieve_cheque(
        account_number,
        cheque_number
    )

    if matched_record is None:
        return {
            "decision": "REJECT",
            "risk_score": 100,
            "matched": False,
            "reasons": [
                "No matching banking record found"
            ]
        }

    account = matched_record["account"]
    cheque = matched_record["cheque"]

    # Run validation checks
    validation_result = validate_cheque(
        extracted_data,
        account,
        cheque
    )

    # Collect validation failures
    mismatches = validation_result["failures"]

    # Calculate fraud risk
    risk_score, fraud_reasons = apply_fraud_rules(
        extracted_data,
        cv_validation,
        account,
        cheque,
        mismatches
    )

    # Make final decision
    decision = make_decision(risk_score)

    return {
        "decision": decision,
        "risk_score": min(risk_score, 100),
        "matched": True,
        "validation": validation_result,
        "reasons": list(dict.fromkeys(fraud_reasons))
    }