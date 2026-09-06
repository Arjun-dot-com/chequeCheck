def make_decision(risk_score):
    if risk_score >= 70:
        return "REJECT"

    if risk_score >= 20:
        return "MANUAL_REVIEW"

    return "APPROVE"