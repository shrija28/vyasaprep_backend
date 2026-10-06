import string
import uuid

import pytest
from flask import Flask, g
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from smartkcet.db.base import Base
from smartkcet.db.models import (
    Exam,
    ExamSet,
    ExamSetQuestion,
    InstitutionBatch,
    Question,
)
from smartkcet.db.subscription_models import Institution
from smartkcet.institution import content
from smartkcet.rag.mcq_extractor import normalize_question_fingerprint


@pytest.fixture
def exam_context(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    app = Flask(__name__)
    institution_ids = [uuid.uuid4(), uuid.uuid4()]
    session.add_all(
        [
            Institution(
                id=institution_id,
                name=f"Test institution {index}",
                contact_phone=f"555000{index}",
            )
            for index, institution_id in enumerate(institution_ids)
        ]
    )
    session.commit()

    def create_exam(institution_id, subject="Biology", batch_id=None):
        monkeypatch.setattr(
            content,
            "require_institution_admin",
            lambda: {"institution_id": str(institution_id)},
        )
        body = {"subject": subject}
        if batch_id:
            body["batch_id"] = str(batch_id)
        with app.test_request_context("/content/exams", method="POST", json=body):
            g.db = session
            result = content.create_institution_exam()
        if isinstance(result, tuple):
            response, status = result
        else:
            response, status = result, result.status_code
        return response, status

    def call_question_endpoint(endpoint, institution_id):
        monkeypatch.setattr(
            content,
            "require_institution_admin",
            lambda: {"institution_id": str(institution_id)},
        )
        with app.test_request_context(endpoint):
            g.db = session
            if endpoint.startswith("/content/questions/counts"):
                return content.get_question_counts()
            return content.list_institution_questions()

    yield session, institution_ids, create_exam, call_question_endpoint

    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def _alpha_token(index):
    alphabet = string.ascii_lowercase
    token = ""
    while True:
        index, remainder = divmod(index, len(alphabet))
        token = alphabet[remainder] + token
        if index == 0:
            return token
        index -= 1


def _add_questions(session, count, institution_id=None, subject="Biology", batch_id=None, stems=None):
    batch_id = batch_id or uuid.uuid4()
    questions = [
        Question(
            subject=subject,
            question_text=(
                stems[index]
                if stems is not None
                else f"Which biological process is represented by unique marker {_alpha_token(uuid.uuid4().int)}?"
            ),
            options=["Option one", "Option two", "Option three", "Option four"],
            correct_option="0",
            topic="General",
            generation_batch_id=batch_id,
            institution_id=institution_id,
        )
        for index in range(count)
    ]
    session.add_all(questions)
    session.flush()
    return questions


def _exam_question_rows(session, exam_id):
    return session.execute(
        select(ExamSet.set_label, Question)
        .join(ExamSetQuestion, ExamSetQuestion.exam_set_id == ExamSet.id)
        .join(Question, Question.id == ExamSetQuestion.question_id)
        .where(ExamSet.exam_id == exam_id)
    ).all()


def _create_and_load(exam_context, institution_id, subject="Biology", batch_id=None):
    session, _, create_exam, _ = exam_context
    response, status = create_exam(institution_id, subject, batch_id)
    assert status == 201, getattr(response, "get_json", lambda: {})()
    return response.get_json(), _exam_question_rows(session, uuid.UUID(response.get_json()["exam_id"]))


def test_institution_questions_are_used_before_platform_questions(exam_context):
    session, institution_ids, _, _ = exam_context
    owned = _add_questions(session, 80, institution_ids[0])
    _add_questions(session, 100)

    _, rows = _create_and_load(exam_context, institution_ids[0])

    assert len(rows) == 4 * 60
    assert all(question.institution_id == institution_ids[0] for _, question in rows)
    assert len({question.id for _, question in rows if question.id in {q.id for q in owned}}) == 60


def test_platform_questions_fill_only_the_remaining_slots(exam_context):
    session, institution_ids, _, _ = exam_context
    owned = _add_questions(session, 25, institution_ids[0])
    platform = _add_questions(session, 100)

    _, rows = _create_and_load(exam_context, institution_ids[0])

    unique_questions = {question.id: question for _, question in rows}
    assert len(unique_questions) == 60
    assert sum(question.institution_id == institution_ids[0] for question in unique_questions.values()) == 25
    assert sum(question.institution_id is None for question in unique_questions.values()) == 35
    assert set(unique_questions).issubset({q.id for q in owned + platform})


def test_combined_shortage_returns_clear_422_without_creating_exam(exam_context):
    session, institution_ids, create_exam, _ = exam_context
    _add_questions(session, 20, institution_ids[0])
    _add_questions(session, 25)

    response, status = create_exam(institution_ids[0])

    assert status == 422
    assert response.get_json() == {
        "error": "insufficient_questions",
        "subject": "Biology",
        "count": 45,
        "required": 60,
        "message": "Not enough eligible questions available for Biology. 45 available, 60 required.",
    }
    assert session.scalar(select(func.count(Exam.id))) == 0


def test_normalized_duplicate_fingerprints_are_not_selected_twice(exam_context):
    session, institution_ids, _, _ = exam_context
    shared = [f"Which biological process is represented by shared marker {_alpha_token(i)}?" for i in range(10)]
    owned = _add_questions(session, 25, institution_ids[0], stems=shared + [
        f"Which biological process is represented by owned marker {_alpha_token(i + 30)}?"
        for i in range(15)
    ])
    platform = _add_questions(session, 50, stems=shared + [
        f"Which biological process is represented by platform marker {_alpha_token(i + 60)}?"
        for i in range(40)
    ])

    _, rows = _create_and_load(exam_context, institution_ids[0])

    selected = {question.id: question for _, question in rows}
    selected_fingerprints = [normalize_question_fingerprint(q.question_text) for q in selected.values()]
    assert len(selected) == 60
    assert len(selected_fingerprints) == len(set(selected_fingerprints))
    assert len(set(selected).intersection({q.id for q in owned})) == 25
    assert len(set(selected).intersection({q.id for q in platform})) == 35


def test_questions_used_by_this_institution_are_excluded(exam_context):
    session, institution_ids, _, _ = exam_context
    used_questions = _add_questions(session, 10, institution_ids[0])
    unused_questions = _add_questions(session, 60, institution_ids[0])
    previous_exam = Exam(subject="Biology", institution_id=institution_ids[0])
    session.add(previous_exam)
    session.flush()
    previous_set = ExamSet(exam_id=previous_exam.id, set_label="A")
    session.add(previous_set)
    session.flush()
    session.add_all(
        [
            ExamSetQuestion(exam_set_id=previous_set.id, question_id=q.id, order_index=index)
            for index, q in enumerate(used_questions)
        ]
    )
    session.commit()

    _, rows = _create_and_load(exam_context, institution_ids[0])

    selected_ids = {question.id for _, question in rows}
    assert selected_ids == {question.id for question in unused_questions}
    assert selected_ids.isdisjoint({question.id for question in used_questions})


def test_platform_questions_can_be_reused_by_another_institution(exam_context):
    _, institution_ids, _, _ = exam_context
    platform = _add_questions(exam_context[0], 60)

    _, rows_a = _create_and_load(exam_context, institution_ids[0])
    _, rows_b = _create_and_load(exam_context, institution_ids[1])

    expected_ids = {question.id for question in platform}
    assert {question.id for _, question in rows_a} == expected_ids
    assert {question.id for _, question in rows_b} == expected_ids


def test_batch_filter_applies_only_to_institution_questions(exam_context):
    session, institution_ids, _, _ = exam_context
    requested_batch = uuid.uuid4()
    other_batch = uuid.uuid4()
    session.add(InstitutionBatch(id=requested_batch, institution_id=institution_ids[0], name="Requested"))
    session.flush()
    owned_requested = _add_questions(session, 25, institution_ids[0], batch_id=requested_batch)
    owned_other = _add_questions(session, 40, institution_ids[0], batch_id=other_batch)
    platform = _add_questions(session, 35, batch_id=other_batch)

    _, rows = _create_and_load(exam_context, institution_ids[0], batch_id=requested_batch)

    selected_ids = {question.id for _, question in rows}
    assert len(selected_ids) == 60
    assert selected_ids.issuperset({question.id for question in owned_requested})
    assert selected_ids.isdisjoint({question.id for question in owned_other})
    assert selected_ids.issuperset({question.id for question in platform})


def test_exam_subject_filter_excludes_other_subject_questions(exam_context):
    session, institution_ids, _, _ = exam_context
    _add_questions(session, 60, subject="Biology")
    _add_questions(session, 100, subject="Physics")

    _, rows = _create_and_load(exam_context, institution_ids[0], subject="Biology")

    assert len(rows) == 4 * 60
    assert {question.subject for _, question in rows} == {"Biology"}


def test_four_sets_have_the_same_60_questions(exam_context):
    session, institution_ids, _, _ = exam_context
    _add_questions(session, 60)

    _, rows = _create_and_load(exam_context, institution_ids[0])

    sets = {}
    for label, question in rows:
        sets.setdefault(label, set()).add(question.id)
    assert set(sets) == {"A", "B", "C", "D"}
    assert all(len(question_ids) == 60 for question_ids in sets.values())
    assert all(question_ids == sets["A"] for question_ids in sets.values())


def test_question_bank_list_and_counts_remain_institution_scoped(exam_context):
    session, institution_ids, _, call_question_endpoint = exam_context
    own_questions = _add_questions(session, 2, institution_ids[0])
    _add_questions(session, 3, institution_ids[1])
    _add_questions(session, 4)

    question_list = call_question_endpoint("/content/questions", institution_ids[0])
    counts = call_question_endpoint("/content/questions/counts", institution_ids[0])

    assert question_list["total"] == 2
    assert {question["id"] for question in question_list["questions"]} == {
        str(question.id) for question in own_questions
    }
    assert counts["counts"]["Biology"] == 2