import unittest
import os
import lale
import urllib
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.neighbors import KNeighborsClassifier as KNN


class TestAutoAIOutputConsumption(unittest.TestCase):

    pickled_model_path = "credit_risk.pickle"
    pickled_model_url = "https://github.com/pmservice/wml-sample-models/raw/master/autoai/credit-risk-prediction/model/credit_risk.pickle"
    train_csv_path = "german_credit_data_biased_training.csv"
    train_csv_url = "https://raw.githubusercontent.com/pmservice/wml-sample-models/master/autoai/credit-risk-prediction/data/german_credit_data_biased_training.csv"
    training_df = None
    model = None
    prefix_model = None
    refined_model = None
    pipeline_content = None
    pp_pipeline: lale.operators.TrainablePipeline = None

    @classmethod
    def setUp(cls) -> None:
        urllib.request.urlretrieve(cls.pickled_model_url, cls.pickled_model_path)
        urllib.request.urlretrieve(cls.train_csv_url, cls.train_csv_path)
        TestAutoAIOutputConsumption.training_df = pd.read_csv(TestAutoAIOutputConsumption.train_csv_path)

    def test_01_load_pickled_model(self):
        try:
            TestAutoAIOutputConsumption.model = joblib.load(TestAutoAIOutputConsumption.pickled_model_path)
            print(type(TestAutoAIOutputConsumption.model))
            print(TestAutoAIOutputConsumption.model)
        except Exception as e:
            assert False, f"Exception was thrown during model pickle: {e}"

    def test_02_predict_on_model(self):
        check_X = TestAutoAIOutputConsumption.training_df.drop(['Risk'], axis=1).values
        x = check_X
        try:
            pred = TestAutoAIOutputConsumption.model.predict(x)
            assert len(pred) == len(x), f"Prediction has returned unexpected number of rows {len(pred)} - expected {len(x)}"
        except Exception as e:
            assert False, f"Exception was thrown during model prediction: {e}"

    def test_03_print_pipeline(self):
        lale_pipeline = TestAutoAIOutputConsumption.model
        TestAutoAIOutputConsumption.pipeline_content = lale_pipeline.pretty_print()
        assert type(TestAutoAIOutputConsumption.pipeline_content) == str
        assert len(TestAutoAIOutputConsumption.pipeline_content) > 0
        assert "lale.wrap_imported_operators()" in TestAutoAIOutputConsumption.pipeline_content

    def test_04_execute_pipeline(self):
        try:
            with open("pp_pipeline.py", "w") as pipeline_f:
                pipeline_f.write(TestAutoAIOutputConsumption.pipeline_content)

            import importlib.util
            import lale.operators

            spec = importlib.util.spec_from_file_location("pp_pipeline", "pp_pipeline.py")
            pipeline_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pipeline_module)
            TestAutoAIOutputConsumption.pp_pipeline = pipeline_module.pipeline
            assert(isinstance(TestAutoAIOutputConsumption.pp_pipeline, lale.operators.TrainablePipeline))
        except Exception as e:
            assert False, f"{e}"
        finally:
            try:
                os.remove("pp_pipeline.py")
            except:
                print("Couldn't remove pp_pipeline.py file")

    def test_05_train_pretty_print_pipeline(self):
        train_X = TestAutoAIOutputConsumption.training_df.drop(['Risk'], axis=1).values
        train_y = TestAutoAIOutputConsumption.training_df.Risk.values

        TestAutoAIOutputConsumption.pp_pipeline.fit(train_X, train_y)

    def test_06_predict_on_pretty_print_pipeline(self):
        check_X = TestAutoAIOutputConsumption.training_df.drop(['Risk'], axis=1).values
        x = check_X
        try:
            pred = TestAutoAIOutputConsumption.pp_pipeline.predict(x)
            assert len(pred) == len(x), f"Prediction on pretty print model has returned unexpected number of rows {len(pred)} - expected {len(x)}"
        except Exception as e:
            assert False, f"Exception was thrown during pretty print model prediction: {e}"

    def test_07_convert_model_to_lale(self):
        import lale.operators
        try:
            lale_pipeline = TestAutoAIOutputConsumption.model
            TestAutoAIOutputConsumption.prefix_model = lale_pipeline.remove_last().freeze_trainable()
            assert(isinstance(TestAutoAIOutputConsumption.prefix_model, lale.operators.TrainablePipeline))
        except Exception as e:
            assert False, f"Exception was thrown during model prediction: {e}"

    def test_08_refine_model_with_lale(self):
        from lale.lib.lale import Hyperopt
        from lale import wrap_imported_operators

        wrap_imported_operators()
        try:
            print(f"prefix_model is of type: {type(TestAutoAIOutputConsumption.prefix_model)}")
            print(f"LR is of type {type(LR)}")
            # This is for classifiers, regressors needs to have different operators & different scoring metrics (e.g 'r2')
            new_model = TestAutoAIOutputConsumption.prefix_model >> (LR | Tree | KNN)
            train_X = TestAutoAIOutputConsumption.training_df.drop(['Risk'], axis=1).values
            train_y = TestAutoAIOutputConsumption.training_df['Risk'].values
            hyperopt = Hyperopt(estimator=new_model, cv=2, max_evals=3, scoring='roc_auc')
            hyperopt_pipelines = hyperopt.fit(train_X, train_y)
            TestAutoAIOutputConsumption.refined_model = hyperopt_pipelines.get_pipeline()
        except Exception as e:
            assert False, f"Exception was thrown during model refinery: {e}"

    def test_09_predict_refined_model(self):
        check_X = TestAutoAIOutputConsumption.training_df.drop(['Risk'], axis=1).values
        x = check_X
        try:
            pred = TestAutoAIOutputConsumption.refined_model.predict(x)
            assert len(pred) == len(
                x), f"Prediction on refined model has returned unexpected number of rows {len(pred)} - expected {len(x)}"
        except Exception as e:
            assert False, f"Exception was thrown during refined model prediction: {e}"
