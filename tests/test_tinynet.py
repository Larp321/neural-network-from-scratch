"""Unit tests (standard-library unittest; also runs under pytest).

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinynet import (MSE, SGD, Adam, BinaryCrossEntropy, CosineDecay, Dense, Dropout,  # noqa: E402
                     LeakyReLU, ReLU, RMSProp, Sequential, Sigmoid, SoftmaxCrossEntropy,
                     StepDecay, Tanh, sigmoid, softmax)
from tinynet import datasets, metrics, viz  # noqa: E402
from tinynet.gradcheck import check_input_gradient, check_model_gradients  # noqa: E402

TOL = 1e-6


class TestGradients(unittest.TestCase):
    """Backprop must match central finite differences for every layer and loss."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(size=(6, 4))

    def _check(self, model, y, loss):
        errors = check_model_gradients(model, self.X, y, loss)
        for name, err in errors.items():
            self.assertLess(err, TOL, f"parameter {name}: relative error {err:.2e}")
        self.assertLess(check_input_gradient(model, self.X, y, loss), TOL)

    def test_every_activation_with_softmax_ce(self):
        y = self.rng.integers(0, 3, 6)
        for act in (Tanh, Sigmoid, lambda: LeakyReLU(0.1), ReLU):
            with self.subTest(act=act):
                r = np.random.default_rng(1)
                model = Sequential([Dense(4, 5, rng=r), act(), Dense(5, 3, rng=r)])
                self._check(model, y, SoftmaxCrossEntropy())

    def test_one_hot_targets_and_label_smoothing(self):
        y = datasets.one_hot(self.rng.integers(0, 3, 6), 3)
        model = Sequential([Dense(4, 5, rng=self.rng), Tanh(), Dense(5, 3, rng=self.rng)])
        self._check(model, y, SoftmaxCrossEntropy(label_smoothing=0.1))

    def test_binary_cross_entropy(self):
        y = self.rng.integers(0, 2, 6)
        model = Sequential([Dense(4, 5, rng=self.rng), Tanh(), Dense(5, 1, rng=self.rng)])
        self._check(model, y, BinaryCrossEntropy())

    def test_mse(self):
        y = self.rng.normal(size=(6, 2))
        model = Sequential([Dense(4, 5, rng=self.rng), Tanh(), Dense(5, 2, rng=self.rng)])
        self._check(model, y, MSE())

    def test_l2_regularization(self):
        y = self.rng.integers(0, 3, 6)
        model = Sequential([Dense(4, 5, rng=self.rng, l2=0.1), Tanh(), Dense(5, 3, rng=self.rng, l2=0.05)])
        self._check(model, y, SoftmaxCrossEntropy())

    def test_deep_network(self):
        y = self.rng.integers(0, 4, 6)
        r = self.rng
        model = Sequential([Dense(4, 8, rng=r), Tanh(), Dense(8, 8, rng=r), Sigmoid(),
                            Dense(8, 8, rng=r), Tanh(), Dense(8, 4, rng=r)])
        self._check(model, y, SoftmaxCrossEntropy())


class TestLayers(unittest.TestCase):
    def test_dense_shapes_and_init(self):
        d = Dense(10, 7, rng=np.random.default_rng(0))
        self.assertEqual(d.forward(np.ones((3, 10))).shape, (3, 7))
        self.assertEqual(d.num_params(), 77)
        self.assertTrue(np.all(d.params["b"] == 0))
        with self.assertRaises(ValueError):
            Dense(2, 2, init="nope")

    def test_stable_sigmoid_and_softmax(self):
        x = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
        s = sigmoid(x)
        self.assertTrue(np.all(np.isfinite(s)))
        np.testing.assert_allclose(s[[0, 2, 4]], [0.0, 0.5, 1.0])
        p = softmax(np.array([[1000.0, 1000.0, -1000.0]]))
        np.testing.assert_allclose(p, [[0.5, 0.5, 0.0]])

    def test_dropout_train_vs_eval(self):
        d = Dropout(0.5, rng=np.random.default_rng(0))
        x = np.ones((1000, 100))
        np.testing.assert_array_equal(d.forward(x, training=False), x)
        out = d.forward(x, training=True)
        self.assertAlmostEqual(out.mean(), 1.0, delta=0.02)  # inverted dropout preserves expectation
        self.assertAlmostEqual((out == 0).mean(), 0.5, delta=0.02)
        grad = d.backward(np.ones_like(x))
        np.testing.assert_array_equal(grad == 0, out == 0)
        with self.assertRaises(ValueError):
            Dropout(1.0)


class TestLosses(unittest.TestCase):
    def test_cross_entropy_known_value(self):
        logits = np.zeros((2, 4))  # uniform -> loss = log(4)
        self.assertAlmostEqual(SoftmaxCrossEntropy()(logits, np.array([0, 3])), np.log(4))

    def test_bce_extreme_logits_are_finite(self):
        loss = BinaryCrossEntropy()(np.array([[1000.0], [-1000.0]]), np.array([0, 1]))
        self.assertAlmostEqual(loss, 1000.0)

    def test_mse_value(self):
        self.assertAlmostEqual(MSE()(np.array([[1.0], [3.0]]), np.array([0.0, 1.0])), 0.5 * (1 + 4) / 2)


class TestOptimizers(unittest.TestCase):
    def _quadratic_descent(self, opt, steps=500):
        """Minimise f(w) = ||w - 3||^2 using a Dense layer's bias as the parameter."""
        layer = Dense(1, 2, rng=np.random.default_rng(0))
        for _ in range(steps):
            layer.grads["W"] = np.zeros_like(layer.params["W"])
            layer.grads["b"] = 2 * (layer.params["b"] - 3.0)
            opt.step([layer])
        return layer.params["b"]

    def test_all_optimizers_converge(self):
        for opt in (SGD(0.1), SGD(0.05, momentum=0.9), SGD(0.05, momentum=0.9, nesterov=True),
                    RMSProp(0.05), Adam(0.1)):
            with self.subTest(opt=repr(opt)):
                np.testing.assert_allclose(self._quadratic_descent(opt), [3.0, 3.0], atol=1e-2)

    def test_gradient_clipping(self):
        layer = Dense(1, 1, rng=np.random.default_rng(0))
        layer.params["b"][:] = 0.0
        layer.grads = {"W": np.zeros((1, 1)), "b": np.array([100.0])}
        SGD(1.0, clip_norm=1.0).step([layer])
        np.testing.assert_allclose(layer.params["b"], [-1.0])

    def test_schedules(self):
        s = StepDecay(1.0, drop=0.5, every=10)
        self.assertEqual([s(0), s(9), s(10), s(25)], [1.0, 1.0, 0.5, 0.25])
        c = CosineDecay(1.0, total_epochs=11, min_lr=0.1)
        self.assertAlmostEqual(c(0), 1.0)
        self.assertAlmostEqual(c(5), 0.55)
        self.assertAlmostEqual(c(10), 0.1)


class TestTraining(unittest.TestCase):
    def test_learns_xor(self):
        rng = np.random.default_rng(0)
        X, y = datasets.make_xor()
        model = Sequential([Dense(2, 8, init="xavier", rng=rng), Tanh(), Dense(8, 1, rng=rng)])
        model.fit(X, y, BinaryCrossEntropy(), SGD(0.5, momentum=0.9), epochs=300, batch_size=4, rng=rng, verbose=0)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_learns_spiral(self):
        rng = np.random.default_rng(1)
        X, y = datasets.make_spiral(100, 3, rng=rng)
        model = Sequential([Dense(2, 64, rng=rng), ReLU(), Dense(64, 3, rng=rng)])
        h = model.fit(X, y, SoftmaxCrossEntropy(), Adam(0.01), epochs=150, batch_size=32, rng=rng, verbose=0)
        self.assertGreater(h["acc"][-1], 0.9)
        self.assertLess(h["loss"][-1], h["loss"][0])

    def test_regression(self):
        rng = np.random.default_rng(2)
        X, y = datasets.make_sine(200, noise=0.05, rng=rng)
        model = Sequential([Dense(1, 32, init="xavier", rng=rng), Tanh(), Dense(32, 1, rng=rng)])
        model.fit(X, y, MSE(), Adam(0.01), epochs=300, batch_size=32, rng=rng, verbose=0)
        rmse = np.sqrt(np.mean((model.forward(X) - np.sin(X)) ** 2))
        self.assertLess(rmse, 0.1)

    def test_early_stopping_restores_best_weights(self):
        rng = np.random.default_rng(3)
        X, y = datasets.make_moons(60, noise=0.4, rng=rng)
        Xv, yv = datasets.make_moons(400, noise=0.4, rng=rng)
        model = Sequential([Dense(2, 128, rng=rng), ReLU(), Dense(128, 2, rng=rng)])
        loss = SoftmaxCrossEntropy()
        h = model.fit(X, y, loss, Adam(0.02), epochs=500, batch_size=8, validation_data=(Xv, yv),
                      early_stopping=10, rng=rng, verbose=0)
        self.assertLess(len(h["loss"]), 500)
        self.assertAlmostEqual(model.evaluate(Xv, yv, loss)[0], min(h["val_loss"]), places=6)

    def test_save_and_load(self):
        rng = np.random.default_rng(4)
        make = lambda r: Sequential([Dense(3, 4, rng=r), ReLU(), Dense(4, 2, rng=r)])  # noqa: E731
        a, b = make(rng), make(np.random.default_rng(99))
        X = rng.normal(size=(5, 3))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.npz")
            a.save(path)
            b.load(path)
        np.testing.assert_allclose(a.forward(X), b.forward(X))

    def test_summary_counts_parameters(self):
        model = Sequential([Dense(2, 3), ReLU(), Dense(3, 1)])
        self.assertEqual(model.num_params(), 2 * 3 + 3 + 3 + 1)
        self.assertIn("13", model.summary())


class TestDataAndMetrics(unittest.TestCase):
    def test_dataset_shapes(self):
        rng = np.random.default_rng(0)
        for X, y, n in [(*datasets.make_spiral(50, 4, rng=rng), 200), (*datasets.make_moons(101, rng=rng), 101),
                        (*datasets.make_circles(80, rng=rng), 80)]:
            self.assertEqual(X.shape, (n, 2))
            self.assertEqual(y.shape, (n,))

    def test_split_and_standardize(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(5, 3, (100, 3)), np.arange(100)
        Xtr, Xte, ytr, yte = datasets.train_test_split(X, y, 0.2, rng=rng)
        self.assertEqual((len(Xtr), len(Xte)), (80, 20))
        self.assertEqual(sorted(np.r_[ytr, yte].tolist()), list(range(100)))
        Z = datasets.Standardizer().fit_transform(Xtr)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(Z.std(0), 1)

    def test_read_idx_format(self):
        import gzip
        import struct
        arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.gz")
            with gzip.open(path, "wb") as f:
                f.write(struct.pack(">HBB", 0, 0x08, 3) + struct.pack(">III", 2, 3, 4) + arr.tobytes())
            np.testing.assert_array_equal(datasets._read_idx(path), arr)

    def test_metrics(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        self.assertAlmostEqual(metrics.accuracy(y_pred, y_true), 4 / 6)
        cm = metrics.confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        p, r, f = metrics.precision_recall_f1(y_true, y_pred)
        np.testing.assert_allclose(r, [0.5, 1.0, 0.5])
        np.testing.assert_allclose(p, [0.5, 2 / 3, 1.0])
        self.assertIn("accuracy", metrics.classification_report(y_true, y_pred))
        # logits and binary-logit inputs
        self.assertEqual(metrics.accuracy(np.array([[2.0, 1.0], [0.0, 3.0]]), np.array([0, 1])), 1.0)
        self.assertEqual(metrics.accuracy(np.array([[2.0], [-1.0]]), np.array([[1], [0]])), 1.0)


class TestHandwriting(unittest.TestCase):
    def test_every_style_renders_mnist_like_images(self):
        from tinynet.handwriting import STYLES, make_styled_digits
        X, y, s = make_styled_digits(2, rng=np.random.default_rng(0), return_styles=True)
        self.assertEqual(X.shape, (2 * len(STYLES), 784))
        self.assertEqual(set(s), set(STYLES))
        self.assertTrue(np.all((X >= 0) & (X <= 1)))
        for img, label, name in zip(X.reshape(-1, 28, 28), y, s):
            self.assertEqual(label, STYLES[name][0])
            self.assertGreater(img.sum(), 10, name)            # visible ink
            ys, xs = np.nonzero(img > 0.1)
            self.assertLessEqual(max(np.ptp(xs), np.ptp(ys)) + 1, 21, name)  # longest side scaled to ~20 px
            yy, xx = np.mgrid[0:28, 0:28]
            com = ((xx * img).sum() / img.sum(), (yy * img).sum() / img.sum())
            self.assertLess(max(abs(com[0] - 14), abs(com[1] - 14)), 1.0, name)  # centred by mass

    def test_preprocess_empty_canvas(self):
        from tinynet.handwriting import CANVAS, preprocess
        self.assertIsNone(preprocess(np.zeros((CANVAS, CANVAS))))


class TestViz(unittest.TestCase):
    def test_png_signature_and_plots(self):
        self.assertTrue(viz.png_bytes(np.zeros((4, 5, 3), np.uint8)).startswith(b"\x89PNG"))
        rng = np.random.default_rng(0)
        X, y = datasets.make_moons(50, rng=rng)
        model = Sequential([Dense(2, 8, rng=rng), ReLU(), Dense(8, 2, rng=rng)])
        with tempfile.TemporaryDirectory() as d:
            viz.line_chart({"a": [3, 2, 1], "b": [1, 0.5, 0.2]}, os.path.join(d, "l.svg"), log_y=True)
            acc = viz.decision_boundary(model, X, y, os.path.join(d, "b.svg"), loss=SoftmaxCrossEntropy(), resolution=20)
            viz.confusion_matrix_plot(np.eye(3, dtype=int), os.path.join(d, "c.svg"))
            for f in ("l.svg", "b.svg", "c.svg"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    self.assertTrue(fh.read().startswith("<svg"))
        self.assertTrue(0.0 <= acc <= 1.0)


if __name__ == "__main__":
    unittest.main()
