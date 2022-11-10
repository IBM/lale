import os
import re
import sys
import time
import traceback
import unittest
import urllib.request
from typing import Optional

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

import lale.operators
from lale.lib.autoai_libs import wrap_pipeline_segments

assert sklearn.__version__ == "0.23.1", "This test is for scikit-learn 0.23.1."


def _println_pos(message):
    tb = traceback.extract_stack()[-2]
    match = re.search(r"<ipython-input-([0-9]+)-", tb[0])
    if match:
        pos = f"notebook cell [{match[1]}] line {tb[1]}"
    else:
        pos = f"{tb[0]}:{tb[1]}"
    strtime = time.strftime("%Y-%m-%d_%H-%M-%S")
    to_log = f"{pos}: {strtime} {message}"

    # if we are running in a notebook, then we also want to print to the console
    # (stored in sys.__stdout__) instead of just the (redirected) sys.stdout
    # that goes only to the notebook
    # This simplifies finding where the notbook ran into a problem when a test fails
    out_file = sys.__stdout__ if match else sys.stdout
    print(to_log, file=out_file)


class TestAutoAIOutputConsumption(unittest.TestCase):

    pickled_model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "lale",
        "datasets",
        "autoai",
        "credit_risk.pickle",
    )
    train_csv_path = "german_credit_data_biased_training.csv"
    train_csv_url = "https://raw.githubusercontent.com/pmservice/wml-sample-models/master/autoai/credit-risk-prediction/data/german_credit_data_biased_training.csv"
    training_df = None
    model = None
    prefix_model = None
    refined_model = None
    pipeline_content: Optional[str] = None
    pp_pipeline: Optional[lale.operators.TrainablePipeline] = None

    @classmethod
    def setUp(cls) -> None:  # pylint:disable=arguments-differ
        urllib.request.urlretrieve(cls.train_csv_url, cls.train_csv_path)
        TestAutoAIOutputConsumption.training_df = pd.read_csv(
            TestAutoAIOutputConsumption.train_csv_path
        )

    def test_01_load_pickled_model(self):
        try:
            old_model = joblib.load(TestAutoAIOutputConsumption.pickled_model_path)
            TestAutoAIOutputConsumption.model = old_model
            _println_pos(f"type(model) {type(TestAutoAIOutputConsumption.model)}")
            _println_pos(f"model {str(TestAutoAIOutputConsumption.model)}")
        except Exception as e:
            assert False, f"Exception was thrown during model pickle: {e}"

    def test_02_predict_on_model(self):
        t_df = TestAutoAIOutputConsumption.training_df
        assert t_df is not None
        check_X = t_df.drop(["Risk"], axis=1).values
        x = check_X
        try:
            m = TestAutoAIOutputConsumption.model
            assert m is not None
            pred = m.predict(x)
            assert len(pred) == len(
                x
            ), f"Prediction has returned unexpected number of rows {len(pred)} - expected {len(x)}"
        except Exception as e:
            assert False, f"Exception was thrown during model prediction: {e}"

    def test_03_print_pipeline(self):
        lale_pipeline = TestAutoAIOutputConsumption.model
        assert lale_pipeline is not None
        wrapped_pipeline = wrap_pipeline_segments(lale_pipeline)
        assert wrapped_pipeline is not None
        TestAutoAIOutputConsumption.pipeline_content = wrapped_pipeline.pretty_print()
        assert isinstance(TestAutoAIOutputConsumption.pipeline_content, str)
        assert len(TestAutoAIOutputConsumption.pipeline_content) > 0
        _println_pos(
            f'pretty-printed """{TestAutoAIOutputConsumption.pipeline_content}"""'
        )
        assert (
            "lale.wrap_imported_operators()"
            in TestAutoAIOutputConsumption.pipeline_content  # pylint:disable=unsupported-membership-test
        )

    def test_04_execute_pipeline(self):
        try:
            with open("pp_pipeline.py", "w", encoding="utf-8") as pipeline_f:
                assert TestAutoAIOutputConsumption.pipeline_content is not None
                pipeline_f.write(TestAutoAIOutputConsumption.pipeline_content)

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "pp_pipeline", "pp_pipeline.py"
            )
            assert spec is not None
            pipeline_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            # the type stubs for _Loader are currently incomplete
            spec.loader.exec_module(pipeline_module)  # type: ignore
            TestAutoAIOutputConsumption.pp_pipeline = pipeline_module.pipeline  # type: ignore
            assert isinstance(
                TestAutoAIOutputConsumption.pp_pipeline,
                lale.operators.TrainablePipeline,
            )
        except Exception as e:
            assert False, f"{e}"
        finally:
            try:
                os.remove("pp_pipeline.py")
            except OSError:
                _println_pos("Couldn't remove pp_pipeline.py file")

    def test_05_train_pretty_print_pipeline(self):
        t_df = TestAutoAIOutputConsumption.training_df
        assert t_df is not None
        train_X = t_df.drop(["Risk"], axis=1).values
        train_y = t_df.Risk.values

        ppp = TestAutoAIOutputConsumption.pp_pipeline
        assert ppp is not None
        ppp.fit(train_X, train_y)

    def test_06_predict_on_pretty_print_pipeline(self):
        t_df = TestAutoAIOutputConsumption.training_df
        assert t_df is not None

        check_X = t_df.drop(["Risk"], axis=1).values
        x = check_X
        try:
            ppp = TestAutoAIOutputConsumption.pp_pipeline
            assert ppp is not None

            pred = ppp.predict(x)
            assert len(pred) == len(
                x
            ), f"Prediction on pretty print model has returned unexpected number of rows {len(pred)} - expected {len(x)}"
        except Exception as e:
            assert (
                False
            ), f"Exception was thrown during pretty print model prediction: {e}"

    def test_07_convert_model_to_lale(self):

        try:
            lale_pipeline = TestAutoAIOutputConsumption.model
            assert lale_pipeline is not None
            TestAutoAIOutputConsumption.prefix_model = (
                lale_pipeline.remove_last().freeze_trainable()
            )
            assert isinstance(
                TestAutoAIOutputConsumption.prefix_model,
                lale.operators.TrainablePipeline,
            )
        except Exception as e:
            assert False, f"Exception was thrown during model prediction: {e}"

    def test_08_refine_model_with_lale(self):
        from lale import wrap_imported_operators
        from lale.lib.lale import Hyperopt

        wrap_imported_operators()
        try:
            _println_pos(
                f"type(prefix_model) {type(TestAutoAIOutputConsumption.prefix_model)}"
            )
            _println_pos(f"type(LR) {type(LR)}")
            # This is for classifiers, regressors needs to have different operators & different scoring metrics (e.g 'r2')
            pm = TestAutoAIOutputConsumption.prefix_model
            assert pm is not None
            new_model = pm >> (LR | Tree | KNN)
            t_df = TestAutoAIOutputConsumption.training_df
            assert t_df is not None
            train_X = t_df.drop(["Risk"], axis=1).values
            train_y = t_df["Risk"].values  # pylint:disable=unsubscriptable-object
            hyperopt = Hyperopt(
                estimator=new_model, cv=2, max_evals=3, scoring="roc_auc"
            )
            hyperopt_pipelines = hyperopt.fit(train_X, train_y)
            TestAutoAIOutputConsumption.refined_model = (
                hyperopt_pipelines.get_pipeline()
            )
        except Exception as e:
            assert False, f"Exception was thrown during model refinery: {e}"

    def test_09_predict_refined_model(self):
        t_df = TestAutoAIOutputConsumption.training_df
        assert t_df is not None
        check_X = t_df.drop(["Risk"], axis=1).values
        x = check_X
        try:
            model = TestAutoAIOutputConsumption.refined_model
            assert model is not None
            pred = model.predict(x)
            assert len(pred) == len(
                x
            ), f"Prediction on refined model has returned unexpected number of rows {len(pred)} - expected {len(x)}"
        except Exception as e:
            assert False, f"Exception was thrown during refined model prediction: {e}"
