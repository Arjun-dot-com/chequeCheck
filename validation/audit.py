import json
from datetime import datetime
from pathlib import Path


AUDIT_FILE = (
    Path(__file__).resolve().parent.parent
    / "bank_data"
    / "audit_log.json"
)


def record_audit(scan_result, validation_result):

    audit_entry = {
        "timestamp": datetime.now().isoformat(),

        "extracted_data":
            scan_result.get(
                "extracted_data",
                {}
            ),

        "validation":
            scan_result.get(
                "validation",
                {}
            ),

        "decision":
            validation_result.get(
                "decision"
            ),

        "risk_score":
            validation_result.get(
                "risk_score"
            ),

        "matched":
            validation_result.get(
                "matched"
            ),

        "validation_passed":
            validation_result.get(
                "validation_passed"
            ),

        "validation_checks":
            validation_result.get(
                "validation_checks",
                {}
            ),

        "reasons":
            validation_result.get(
                "reasons",
                []
            )
    }


    # Create bank_data folder if necessary
    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Read existing records
    if AUDIT_FILE.exists():

        try:

            with open(
                AUDIT_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                audit_records = json.load(file)


            if not isinstance(
                audit_records,
                list
            ):

                audit_records = []


        except (
            json.JSONDecodeError,
            OSError
        ):

            audit_records = []


    else:

        audit_records = []


    # Add new record
    audit_records.append(
        audit_entry
    )


    # Save audit log
    with open(
        AUDIT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            audit_records,
            file,
            indent=2
        )