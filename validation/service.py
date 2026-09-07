from .retriever import retrieve_cheque
from .validator import validate_cheque
from .fraud_rules import apply_fraud_rules
from .decision_engine import make_decision
from .audit import record_audit


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

    # ---------------------------------------------------------
    # Missing cheque number
    # ---------------------------------------------------------
    if not cheque_number:
        result = {
            "decision": "REJECT",
            "risk_score": 100,
            "matched": False,
            "reasons": [
                "Cheque number could not be extracted"
            ]
        }

        record_audit(scan_result, result)
        return result

    # ---------------------------------------------------------
    # Missing account number
    # ---------------------------------------------------------
    if not account_number:
        result = {
            "decision": "REJECT",
            "risk_score": 100,
            "matched": False,
            "reasons": [
                "Account identifier could not be extracted"
            ]
        }

        record_audit(scan_result, result)
        return result

    # ---------------------------------------------------------
    # Retrieve banking record
    # ---------------------------------------------------------
    matched_record = retrieve_cheque(
        account_number,
        cheque_number
    )

    # ---------------------------------------------------------
    # No banking record found
    # ---------------------------------------------------------
    if matched_record is None:
        result = {
            "decision": "REJECT",
            "risk_score": 100,
            "matched": False,
            "reasons": [
                "No matching banking record found"
            ]
        }

        record_audit(scan_result, result)
        return result

    account = matched_record["account"]
    cheque = matched_record["cheque"]

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    validation_result = validate_cheque(
        extracted_data,
        account,
        cheque
    )

    mismatches = validation_result["failures"]

    # ---------------------------------------------------------
    # Fraud detection
    # ---------------------------------------------------------
    risk_score, fraud_reasons = apply_fraud_rules(
        extracted_data,
        cv_validation,
        account,
        cheque,
        mismatches
    )

    # ---------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------
    decision = make_decision(risk_score)

    result = {
        "decision": decision,
        "risk_score": min(risk_score, 100),
        "matched": True,
        "validation_passed": validation_result["passed"],
        "validation_checks": validation_result["checks"],
        "validation_failures": validation_result["failures"],
        "reasons": list(dict.fromkeys(fraud_reasons))
    }

    # ---------------------------------------------------------
    # Audit logging
    # ---------------------------------------------------------
    record_audit(
        scan_result,
        result
    )

    return result