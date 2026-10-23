"""Check agreed examples and the boundaries that used to conflict."""

import math
import unittest

from astrbot_plugin_reply_gate.activity_gate import ReplyAccumulator, validate_activity


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.gate = ReplyAccumulator()
        self.key = ("platform:group", "conversation")

    def test_agreed_repeat_counts_at_different_activities(self):
        scores = [0.1, 0.3, 0.55, 0.6, 0.9]
        expected = {
            0: [None, None, None, None, None],
            0.01: [None, None, None, None, None],
            0.2: [None, None, None, None, 2],
            0.5: [None, None, 5, 3, 1],
            0.6: [None, None, 2, 2, 1],
            0.8: [None, 2, 1, 1, 1],
            0.99: [1, 1, 1, 1, 1],
            1: [1, 1, 1, 1, 1],
        }
        for activity, counts in expected.items():
            for score, count in zip(scores, counts, strict=True):
                with self.subTest(activity=activity, score=score):
                    self.gate.reset(self.key)
                    results = [
                        self.gate.evaluate(self.key, score, activity)
                        for _ in range(count or 100)
                    ]
                    self.assertEqual(
                        [r.reply for r in results],
                        [False] * (len(results) - bool(count))
                        + ([True] if count else []),
                    )

    def test_low_score_consumes_credit_without_negative_debt(self):
        results = [self.gate.evaluate(self.key, p, 0.6) for p in [0.6, 0.1, 0.6, 0.6]]
        self.assertEqual([r.reply for r in results], [False, False, False, True])
        self.assertAlmostEqual(results[0].remaining, 0.8109302162)
        self.assertEqual(results[1].remaining, 0)
        self.assertEqual(results[-1].remaining, 0)
        self.assertAlmostEqual(results[2].remaining, results[0].remaining)

    def test_endpoints_are_finite_and_activity_has_precedence(self):
        for activity in [0, 0.01, 0.5, 0.99, 1]:
            for score in [0, 1]:
                with self.subTest(activity=activity, score=score):
                    result = self.gate.evaluate(self.key, score, activity)
                    self.assertIs(
                        result.reply, activity == 1 or (activity > 0 and score == 1)
                    )
                    self.assertEqual(result.remaining, 0)
                    self.assertIsNone(result.change)

    def test_stronger_endpoint_scores_have_larger_effects(self):
        changes = []
        for score in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            self.gate.reset(self.key)
            changes.append(self.gate.evaluate(self.key, score, 0.5).change)
        self.assertEqual(changes, sorted(changes))
        self.assertGreater(changes[-1] - changes[-2], changes[-2] - changes[-3])
        self.assertGreater(abs(changes[0]), abs(changes[1]))

    def test_neutral_scores_do_not_drift_and_high_scores_do_not_bank_extra_replies(
        self,
    ):
        for _ in range(100):
            self.assertEqual(self.gate.evaluate(self.key, 0.4, 0.6).remaining, 0)
        self.assertTrue(self.gate.evaluate(self.key, 0.999, 0.6).reply)
        self.assertFalse(self.gate.evaluate(self.key, 0.6, 0.6).reply)

    def test_groups_conversations_and_changed_settings_start_independently(self):
        self.assertFalse(self.gate.evaluate(self.key, 0.6, 0.6).reply)
        self.assertFalse(
            self.gate.evaluate(("other:group", "conversation"), 0.6, 0.6).reply
        )
        self.assertFalse(
            self.gate.evaluate((self.key[0], "new-conversation"), 0.6, 0.6).reply
        )
        # The old credit must not cause a reply after changing activity.
        result = self.gate.evaluate(self.key, 0.6, 0.5)
        self.assertFalse(result.reply)
        self.assertEqual(result.before, 0)

    def test_invalid_settings_and_scores_are_rejected(self):
        for value in [True, None, "invalid", -0.1, 1.1, math.inf, math.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_activity(value)
        for value in [True, "0.9", None, -0.1, 1.1, math.inf, math.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate.evaluate(self.key, value, 0.6)
