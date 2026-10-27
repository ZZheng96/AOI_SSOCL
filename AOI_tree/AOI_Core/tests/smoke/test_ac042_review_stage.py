"""AC-042：复核阶段契约的最小回归测试，不触碰生产数据库。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.api import routes_review


class _Session:
    def __init__(self, det, feedback=False):
        self.det = det
        self.feedback = feedback

    def get(self, model, detection_id):
        return self.det

    def query(self, model):
        return self

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return object() if self.feedback else None


class _Scope:
    def __init__(self, sessions):
        self.sessions = iter(sessions)

    def __call__(self):
        session = next(self.sessions)
        class Context:
            def __enter__(inner):
                return session

            def __exit__(inner, *args):
                return False
        return Context()


class ReviewStageTest(unittest.TestCase):
    def _det(self, n_tiles=None):
        return SimpleNamespace(
            id=1, n_tiles=n_tiles or {"decision": "gray"}, category="cat",
            image_path="x.png", final_score=0.2, is_anomaly=False,
        )

    def test_legacy_queue_submit_transition(self):
        det = self._det()
        scope = _Scope([_Session(det), _Session(det)])
        with patch.object(routes_review, "session_scope", scope), \
             patch.object(routes_review.sl_service, "submit_feedback_record",
                          return_value={"feedback_id": 7, "engine_update": None,
                                        "note": "ok", "category": "cat", "pre": None}), \
             patch.object(routes_review, "log_action"):
            out = routes_review.api_review_submit(
                1, routes_review.ReviewSubmitRequest(label=0))
        self.assertEqual(out["review_stage"], "submitted")
        self.assertEqual(det.n_tiles["review_stage"], "submitted")

    def test_duplicate_submission_is_rejected(self):
        det = self._det()
        scope = _Scope([_Session(det, feedback=True)])
        with patch.object(routes_review, "session_scope", scope):
            with self.assertRaises(routes_review.HTTPException) as ctx:
                routes_review.api_review_submit(
                    1, routes_review.ReviewSubmitRequest(label=0))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_unknown_stage_fails_closed(self):
        det = self._det({"decision": "gray", "review_stage": "tampered"})
        scope = _Scope([_Session(det)])
        with patch.object(routes_review, "session_scope", scope):
            with self.assertRaises(routes_review.HTTPException) as ctx:
                routes_review.api_review_submit(
                    1, routes_review.ReviewSubmitRequest(label=0))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_submit_stage_must_be_queued_when_explicit(self):
        det = self._det()
        scope = _Scope([_Session(det)])
        with patch.object(routes_review, "session_scope", scope):
            with self.assertRaises(routes_review.HTTPException) as ctx:
                routes_review.api_review_submit(
                    1, routes_review.ReviewSubmitRequest(
                        label=0, review_stage="submitted"))
        self.assertEqual(ctx.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main(verbosity=2)
