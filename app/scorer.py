def compute_score(checks: dict) -> dict:
    TIMEOUT = 7.0

    results = {}
    total_weight = 0.0
    weighted_score = 0.0
    timeout_penalty = 0.0

    challenge_weights = {
        "BLINK":      0.35,
        "TURN_LEFT":  0.30,
        "TURN_RIGHT": 0.30,
    }

    for key, weight in challenge_weights.items():
        check = checks.get(key, {})
        passed = check.get("passed", False)
        timed_out = check.get("timed_out", False)
        elapsed = check.get("elapsed", None)

        if passed and elapsed is not None:
            # scale from 1.0 (instant) down to 0.6 (just before timeout)
            speed_ratio = 1.0 - (elapsed / TIMEOUT) * 0.4
            check_score = speed_ratio
        elif passed:
            check_score = 0.6
        else:
            check_score = 0.0

        if timed_out:
            timeout_penalty += 0.05

        results[key] = {
            "passed": passed,
            "check_score": round(check_score, 3),
            "elapsed": elapsed
        }

        weighted_score += weight * check_score
        total_weight += weight

    # consistency is a small modifier, not additive weight
    consistency = checks.get("consistency", 1.0)
    consistency_modifier = 0.05 * consistency  # max 0.05 bonus

    raw = weighted_score + consistency_modifier - timeout_penalty
    score = round(max(0.0, min(1.0, raw)), 3)
    status = "live" if score >= 0.60 else "spoof"

    return {
        "status": status,
        "score": score,
        "breakdown": {
            "weighted_score": round(weighted_score, 3),
            "consistency_bonus": round(consistency_modifier, 3),
            "timeout_penalty": round(timeout_penalty, 3),
        },
        "checks": {k: v["passed"] for k, v in results.items()}
    }