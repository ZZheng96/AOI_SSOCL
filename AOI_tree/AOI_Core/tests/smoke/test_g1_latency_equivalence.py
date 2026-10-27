"""G1 时延优化等价性：向量化森林 == sklearn predict_proba；预取 future 与串行结果一致。"""
import pickle
import unittest

import numpy as np

from algo.decision.defect_classifier import DefectClassifier, FIVE


def _fit(n_classes, seed=0, n_feat=430, n=60):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    y = rng.choice(np.arange(n_classes) * (len(FIVE) // n_classes), size=n)
    c = DefectClassifier()
    c.clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=1).fit(X, y)
    c.fitted = True
    return c, rng


class FastForestEquivalenceTest(unittest.TestCase):
    def _check(self, n_classes):
        c, rng = _fit(n_classes)
        for _ in range(40):
            x = rng.normal(size=430).astype(np.float32)
            ref = c.clf.predict_proba(x[None, :])[0]
            got = c._fast_proba(x)
            if got is None:
                self.fail("fast path disabled")
            self.assertLessEqual(float(np.max(np.abs(got - ref))), 1e-9)
            self.assertEqual(int(np.argmax(got)), int(np.argmax(ref)))
        self.assertIsNot(c._packed, False)

    def test_all_classes(self):
        self._check(len(FIVE))

    def test_missing_classes_maps_label(self):
        c, rng = _fit(2)
        x = rng.normal(size=430).astype(np.float32)
        ref = c.clf.predict_proba(x[None, :])[0]
        top = int(np.argmax(ref))
        c._fast_proba(x)
        p = c._fast_proba(x)
        self.assertEqual(FIVE[int(c.clf.classes_[top])],
                         FIVE[int(c.clf.classes_[int(np.argmax(p))])])
        self._check(2)

    def test_pickle_drops_cache(self):
        c, rng = _fit(len(FIVE))
        c._fast_proba(rng.normal(size=430).astype(np.float32))
        c2 = pickle.loads(pickle.dumps(c))
        self.assertFalse(hasattr(c2, "_packed"))


if __name__ == "__main__":
    unittest.main()
