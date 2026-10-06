"""Pure scoring helper for student submissions.

Implements the scoring contract that previously lived inline in
``backend/app.py``'s ``/analyze`` endpoint and is now also wired into
:mod:`smartkcet.routes.legacy`.  Both call sites use this single
function so behaviour stays identical between the legacy ``/analyze``
endpoint and the new role-scoped ``POST /api/student/submit`` endpoint
(task 8.1, REQ-9.3).

The function is intentionally pure: no DB, no side effects, no
network.  Persistence and idempotency are handled by the caller.

Input shape
-----------

``questions`` is a list of dicts with the following keys (extras are
ignored)::

    {
        "q":     str,                # question text
        "opts":  list[str],          # 4 MCQ options
        "ans":   int | str,          # canonical correct option (index or label)
        "topic": str,                # optional, defaults to "General"
        "marks": int,                # optional, defaults to 1
    }

``answers`` is a mapping from the question's stringified index to the
student's selected option (``"0"|"1"|"2"|"3"`` for MCQs, or ``""`` for
unanswered).  This matches the legacy ``ES.answers`` map produced by
``frontend/js/exam.js``.

Output shape
------------

The return value matches the legacy ``/analyze`` body verbatim, plus an
explicit ``topic_breakdown`` alias used by ``Submission.topic_breakdown``
so the persistence call site does not have to remember which of the
two equivalent keys to write::

    {
        "percentage":      int,
        "earned":          int,
        "total":           int,
        "topicScores":     dict[str, {"earned": int, "total": int}],
        "topic_breakdown": dict[str, {"earned": int, "total": int}],   # alias
        "typeScores":      dict[str, {"earned": int, "total": int}],
        "strong":          list[{"topic": str, "pct": int}],
        "canImprove":      list[{"topic": str, "pct": int}],
        "weak":            list[{"topic": str, "pct": int}],
        "questionResults": list[ ... per-question record ... ],
        "pass":            bool,
        "recommendation":  str,
    }

The 70 / 40 thresholds (strong vs. improve vs. weak) and the "Excellent /
Good / Needs improvement" recommendation prefixes mirror the legacy
behaviour and the contract documented in REQ-10.5.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


# Bucket thresholds (REQ-10.5).
_STRONG_THRESHOLD = 70
_IMPROVE_THRESHOLD = 40

# Legacy /analyze pass mark.  Unchanged from app.py so the dashboard
# treats existing and new submissions identically.
_PASS_THRESHOLD = 40


def _pct(earned: int, total: int) -> float:
    """Round earned/total to a percentage, returning 0.0 when total is 0."""

    if total <= 0:
        return 0.0
    return round((earned / total) * 100, 2)


def _normalised_marks(value: Any)-> int:
    """Coerce a question's ``marks`` field to a positive int (default 1)."""

    if isinstance(value, bool):  # bool is a subclass of int — exclude.
        return 1
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return 1


def _resolve_option_index(val: Any, opts: Any = None) -> tuple[Optional[int], Optional[str]]:
    """Resolve an option value (index "0".."3", letter "A".."D", or option text) to (index, clean_text).

    Resolution order (critical for correctness):
      1. Single digit "0"–"3" or single letter "a"–"d" / "A"–"D" → treat as index directly.
         This MUST come before text matching because correct_option is stored as a digit
         string ("0", "1", "2", "3") in the DB, and option texts may themselves be digits
         (e.g. chemistry options "1", "2", "4", "8") which would otherwise cause a false
         text match at the wrong position.
      2. Multi-character value → try to match against cleaned option text.
    """
    if val is None:
        return None, None
    val_str = str(val).strip()
    if not val_str:
        return None, None

    clean_opts: list[str] = []
    if opts and isinstance(opts, (list, tuple)):
        import re
        clean_opts = [
            re.sub(r"^\s*(?:\([A-Da-d1-4]\)|[A-Da-d1-4]\s*[.):\-]|option\s+[A-Da-d1-4]\s*[:\-]?)\s*", "", str(opt), flags=re.IGNORECASE).strip()
            for opt in opts
        ]

    val_lower = val_str.lower()

    # 1. Single digit or single letter → resolve as index FIRST.
    #    Never fall through to text matching for these, because the stored
    #    correct_option is always a digit "0"–"3" and must not be confused
    #    with option content that happens to contain the same character.
    letter_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "0": 0, "1": 1, "2": 2, "3": 3}
    if val_lower in letter_to_idx:
        idx = letter_to_idx[val_lower]
        text = clean_opts[idx].lower() if clean_opts and 0 <= idx < len(clean_opts) else None
        return idx, text

    # 2. Multi-character value → option text match (used when the student's
    #    answer is sent as the full option string rather than an index).
    if clean_opts:
        for i, opt in enumerate(clean_opts):
            if opt.lower() == val_lower:
                return i, opt.lower()

    return None, val_lower


def _is_correct_answer(given: Any, ans: Any, opts: Any = None) -> bool:
    """Robustly check if student's given answer matches the correct option."""
    if given is None or str(given).strip() == "":
        return False

    given_str = str(given).strip()
    ans_str = str(ans).strip() if ans is not None else ""

    if given_str.lower() == ans_str.lower():
        return True

    given_idx, given_text = _resolve_option_index(given, opts)
    ans_idx, ans_text = _resolve_option_index(ans, opts)

    # 1. Index match (0..3)
    if given_idx is not None and ans_idx is not None:
        return given_idx == ans_idx

    # 2. Option text match
    if given_text and ans_text and given_text == ans_text:
        return True

    return False


def score_submission(questions: Iterable[Mapping[str, Any]], answers: Mapping[str, Any])-> dict[str, Any]:
    """Score one submission and return the full result envelope.

    See module docstring for input/output shape.
    """

    questions_list = list(questions)
    total = 0
    earned = 0
    topic_scores: dict[str, dict[str, int]] = {}
    type_scores: dict[str, dict[str, int]] = {}
    question_results: list[dict[str, Any]] = []

    for i, q in enumerate(questions_list):
        marks = _normalised_marks(q.get("marks"))
        total += marks

        topic = q.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            topic = "General"

        topic_scores.setdefault(topic, {"earned": 0, "total": 0})["total"] += marks
        type_scores.setdefault("MCQ", {"earned": 0, "total": 0})["total"] += marks

        given = answers.get(str(i))

        question_earned = 0
        if given is None or str(given).strip() == "":
            status = "unanswered"
        elif _is_correct_answer(given, q.get("ans"), q.get("opts")):
            question_earned = marks
            status = "correct"
        else:
            status = "wrong"

        earned += question_earned
        topic_scores[topic]["earned"] += question_earned
        type_scores["MCQ"]["earned"] += question_earned

        question_results.append(
            {
                "q": q.get("q"),
                "type": "MCQ",
                "topic": topic,
                "given": given,
                "correctAns": q.get("ans"),
                "earned": question_earned,
                "marks": marks,
                "status": status,
            }
        )

    percentage = _pct(earned, total)

    strong: list[dict[str, Any]] = []
    can_improve: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    for topic_name, scores in topic_scores.items():
        topic_pct = _pct(scores["earned"], scores["total"])
        bucket = (
            strong
            if topic_pct >= _STRONG_THRESHOLD
            else can_improve
            if topic_pct >= _IMPROVE_THRESHOLD
            else weak
        )
        bucket.append({"topic": topic_name, "pct": topic_pct})

    if percentage >= 75:
        recommendation = "Excellent! "
    elif percentage >= 50:
        recommendation = "Good effort. "
    else:
        recommendation = "Needs improvement. "
    if weak:
        recommendation += f"Focus on: {', '.join(w['topic'] for w in weak)}."

    return {
        "percentage": percentage,
        "earned": earned,
        "total": total,
        "topicScores": topic_scores,
        # Alias for callers (notably ``Submission.topic_breakdown``) that
        # want a name aligned with the DB column.  The two keys point at
        # the same dict so updates stay in sync — but because the dict is
        # produced fresh per call, persistence callers can safely pass
        # either one to ``json.dumps`` without aliasing concerns.
        "topic_breakdown": topic_scores,
        "typeScores": type_scores,
        "strong": strong,
        "canImprove": can_improve,
        "weak": weak,
        "questionResults": question_results,
        "pass": percentage >= _PASS_THRESHOLD,
        "recommendation": recommendation,
    }


__all__ = ["score_submission"]
