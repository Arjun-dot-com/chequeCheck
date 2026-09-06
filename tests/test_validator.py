from validation.service import process_validation


scan_result = {
    "extracted_data": {
        "cheque_number": {
            "value": "123456",
            "confidence": 97
        },
        "payee": {
            "value": "ABC Traders",
            "confidence": 94
        },
        "amount": {
            "value": "25000",
            "confidence": 96
        },
        "date": {
            "value": "2026-09-05",
            "confidence": 92
        },
        "micr_code": {
            "value": "9876543210",
            "confidence": 95
        },
        "bank_name": {
            "value": "Demo Bank",
            "confidence": 90
        }
    },
    "validation": {
        "is_signed": True,
        "amount_tamper_flag": False,
        "payee_tamper_flag": False
    }
}


result = process_validation(scan_result)


print("\nFINAL DECISION")
print("==============")
print("Decision:", result["decision"])
print("Risk Score:", result["risk_score"])
print("Banking Record Matched:", result["matched"])

print("\nReasons:")

if result["reasons"]:
    for reason in result["reasons"]:
        print("-", reason)
else:
    print("- No fraud indicators detected")