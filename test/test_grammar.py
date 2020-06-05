import unittest
from lale.grammar import Grammar
from lale.operators import make_choice, PlannedOperator, PlannedPipeline, TrainedOperator
from lale import wrap_imported_operators

from lale.lib.sklearn import LogisticRegression as LR
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import PCA
from lale.lib.sklearn import StandardScaler as Scaler
from lale.lib.sklearn import AdaBoostClassifier as Boost
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.lale import NoOp

from lale.lib.lale import Hyperopt
import lale.datasets

class TestGrammar(unittest.TestCase):
    def setUp(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = lale.datasets.load_iris_df()

    def test_grammar_simple(self):
        g = Grammar()
        g.start       = g.estimator
        g.estimator   = (NoOp | g.transformer) >> g.prim_est
        g.transformer = (NoOp | g.transformer) >> g.prim_tfm

        g.prim_est    = LR | KNN
        g.prim_tfm    = PCA | Scaler

        generated = g.unfold(6)
        sample = g.sample(6)
        
        # unfold and sample return a PlannedOperator
        assert isinstance(generated, PlannedOperator)
        assert isinstance(sample, PlannedOperator)
        
        # test getter for methods other than Nonterminal   
        if isinstance(generated, PlannedPipeline):
            assert (generated._name.startswith('pipeline'))
            
        # Train
        gtrainer = Hyperopt(estimator=generated, cv=2, max_evals=6, scoring='r2')
        gtrained = gtrainer.fit(self.train_X, self.train_y)
        assert isinstance(gtrained.get_pipeline(), TrainedOperator)
        
        strainer = Hyperopt(estimator=sample, cv=2, max_evals=6, scoring='r2')
        strained = strainer.fit(self.train_X, self.train_y)
        assert isinstance(strained.get_pipeline(), TrainedOperator)
        
        
        
    def test_grammar_all_combinator(self):
        g = Grammar()

        g.start       = g.estimator
        g.estimator   = g.term_est | g.transformer >> g.term_est
        g.term_est    = g.prim_est | g.ensemble
        g.ensemble    = Boost ( base_estimator = LR )
        g.transformer = g.union_tfm | g.union_tfm >> g.transformer
        g.union_tfm   = g.prim_tfm | g.union_body >> Concat
        g.union_body  = g.transformer | g.transformer & g.union_body
        
        g.prim_est    = LR | KNN
        g.prim_tfm    = PCA | Scaler
        g.ensembler   = Boost

        generated = g.unfold(7)
        sample = g.sample(7)
        assert isinstance(generated, PlannedOperator)
        assert isinstance(sample, PlannedOperator)
        
        # Train
        gtrainer = Hyperopt(estimator=generated, cv=2, max_evals=6, scoring='r2')
        gtrained = gtrainer.fit(self.train_X, self.train_y)
        assert isinstance(gtrained.get_pipeline(), TrainedOperator)
        
        strainer = Hyperopt(estimator=sample, cv=2, max_evals=6, scoring='r2')
        strained = strainer.fit(self.train_X, self.train_y)
        assert isinstance(strained.get_pipeline(), TrainedOperator)