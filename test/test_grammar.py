import unittest

import lale.datasets
from lale.grammar import Grammar
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.lale import Hyperopt, NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import AdaBoostClassifier as Boost
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import LogisticRegression as LR
from lale.lib.sklearn import StandardScaler as Scaler
from lale.operators import PlannedOperator, PlannedPipeline, TrainedOperator


class TestGrammar(unittest.TestCase):
    def setUp(self):
        (
            (self.train_X, self.train_y),
            (self.test_X, self.test_y),
        ) = lale.datasets.load_iris_df()

    def test_grammar_simple(self):
        g = Grammar()
        g.start = g.estimator
        g.estimator = (NoOp | g.transformer) >> g.prim_est
        g.transformer = (NoOp | g.transformer) >> g.prim_tfm

        g.prim_est = LR | KNN
        g.prim_tfm = PCA | Scaler

        generated = g.unfold(6)
        sample = g.sample(6)

        # unfold and sample return a PlannedOperator
        assert isinstance(generated, PlannedOperator)
        assert isinstance(sample, PlannedOperator)

        # test getter for methods other than Nonterminal
        if isinstance(generated, PlannedPipeline):
            assert generated._name.startswith("pipeline")

        try:
            gtrainer = Hyperopt(estimator=generated, max_evals=3, scoring="r2")
            gtrained = gtrainer.fit(self.train_X, self.train_y)
            assert isinstance(gtrained.get_pipeline(), TrainedOperator)
        except ValueError:
            # None of the trials succeeded
            pass

        try:
            strainer = Hyperopt(estimator=sample, max_evals=3, scoring="r2")
            strained = strainer.fit(self.train_X, self.train_y)
            assert isinstance(strained.get_pipeline(), TrainedOperator)
        except ValueError:
            # None of the trials succeeded
            pass

    def test_grammar_all_combinator(self):
        g = Grammar()

        g.start = g.estimator
        g.estimator = g.term_est | g.transformer >> g.term_est
        g.term_est = g.prim_est | g.ensemble
        g.ensemble = Boost(base_estimator=LR)
        g.transformer = g.union_tfm | g.union_tfm >> g.transformer
        g.union_tfm = g.prim_tfm | g.union_body >> Concat
        g.union_body = g.transformer | g.transformer & g.union_body

        g.prim_est = LR | KNN
        g.prim_tfm = PCA | Scaler
        g.ensembler = Boost

        generated = g.unfold(7)
        sample = g.sample(7)
        assert isinstance(generated, PlannedOperator)
        assert isinstance(sample, PlannedOperator)

        # Train
        try:
            gtrainer = Hyperopt(estimator=generated, max_evals=3, scoring="r2")
            gtrained = gtrainer.fit(self.train_X, self.train_y)
            assert isinstance(gtrained.get_pipeline(), TrainedOperator)
        except ValueError:
            # None of the trials succeeded
            pass

        try:
            strainer = Hyperopt(estimator=sample, max_evals=3, scoring="r2")
            strained = strainer.fit(self.train_X, self.train_y)
            assert isinstance(strained.get_pipeline(), TrainedOperator)
        except ValueError:
            # None of the trials succeeded
            pass
