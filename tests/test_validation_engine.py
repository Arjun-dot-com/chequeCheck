from validation.service import process_validation


def create_scan_result(
    cheque_number="123456",
    payee="ABC Traders",
    amount="25000",
    date="2026-09-05",
    account_number="9876543210",
    signed=True,
    amount_tamper=False,
    payee_tamper=False,
    amount_confidence=96,
):
    return {
        "extracted_data": {
            "cheque_number": {
                "value": cheque_number,
                "confidence": 97
            },
            "payee": {
                "value": payee,
                "confidence": 94
            },
            "amount": {
                "value": amount,
                "confidence": amount_confidence
            },
            "date": {
                "value": date,
                "confidence": 92
            },
            "micr_code": {
                "value": account_number,
                "confidence": 95
            },
            "bank_name": {
                "value": "Demo Bank",
                "confidence": 90
            }
        },
        "validation": {
            "is_signed": signed,
            "amount_tamper_flag": amount_tamper,
            "payee_tamper_flag": payee_tamper
        }
    }


def test_valid_cheque_is_approved():
    result = process_validation(create_scan_result())

    assert result["decision"] == "APPROVE"
    assert result["risk_score"] == 0
    assert result["matched"] is True


def test_amount_mismatch_requires_manual_review():
    result = process_validation(
        create_scan_result(amount="75000")
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["risk_score"] == 40


def test_unknown_cheque_is_rejected():
    result = process_validation(
        create_scan_result(cheque_number="999999")
    )

    assert result["decision"] == "REJECT"
    assert result["risk_score"] == 100
    assert result["matched"] is False


def test_missing_cheque_number_is_rejected():
    result = process_validation(
        create_scan_result(cheque_number="")
    )

    assert result["decision"] == "REJECT"
    assert result["risk_score"] == 100


def test_missing_account_identifier_is_rejected():
    result = process_validation(
        create_scan_result(account_number="")
    )

    assert result["decision"] == "REJECT"
    assert result["risk_score"] == 100


def test_missing_signature_requires_manual_review():
    result = process_validation(
        create_scan_result(signed=False)
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["risk_score"] == 40


def test_amount_tampering_requires_manual_review():
    result = process_validation(
        create_scan_result(amount_tamper=True)
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["risk_score"] == 50


def test_payee_tampering_requires_manual_review():
    result = process_validation(
        create_scan_result(payee_tamper=True)
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["risk_score"] == 40


def test_date_mismatch_requires_manual_review():
    result = process_validation(
        create_scan_result(date="2026-09-10")
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["risk_score"] == 20


def test_low_ocr_confidence_requires_manual_review():
    result = process_validation(
        create_scan_result(amount_confidence=20)
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["risk_score"] == 30