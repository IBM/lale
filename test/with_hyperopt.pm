# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.datasets import load_iris
import sklearn.utils
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import importlib
from lale.operators import make_pipeline
from lale.helpers import create_instance_from_hyperopt_search_space
from lale.search.op2hp import hyperopt_search_space

SEED = 3141

class HyperoptClassifierTest():

    def __init__(self, classifier_search_space):
        self.search_space = hp.choice('classifier', [classifier_search_space])
        iris = sklearn.datasets.load_iris()
        X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target)
        self.X_train, self.y_train = X_all[5:], y_all[5:]
        self.X_test, self.y_test = X_all[:5], y_all[:5]

    def hyperopt_train_test(self, params):
        t = params['name']
        del params['name']
        clf = self.get_classifier(t, params)
        #Use the accuracy based on a random train-test split as the evaluation criterion
        X_train_part, X_validation, y_train_part, y_validation = train_test_split(self.X_train, self.y_train, test_size=0.20)
        clf_trained = clf.fit(X_train_part, y_train_part)
        predictions = clf_trained.predict(X_validation)
        accuracy = accuracy_score(y_validation, [round(pred) for pred in predictions])
        return accuracy

    def eval_on_best(self, params):
        t = params['name']
        del params['name']
        clf = self.get_classifier(t, params)
        clf_trained = clf.fit(self.X_train, self.y_train)
        predictions = clf_trained.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, [round(pred) for pred in predictions])
        return accuracy

    def get_classifier(self, classifier_name, param_dict):
        instance = None
        classifier_name_parts = classifier_name.split(".")
        assert(len(classifier_name_parts)) >1, "The classifier name needs to be fully qualified, i.e. module name + class name"
        module_name = ".".join(classifier_name_parts[0:-1])
        class_name = classifier_name_parts[-1]

        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        if param_dict is None:
            instance = class_()
        else:
            instance = class_(**param_dict)
        return instance

    def set_search_space(self, classifier_search_space):
        self.search_space = hp.choice('classifier', [classifier_search_space])

    def test_classifier(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        def f(params):
            acc = self.hyperopt_train_test(params.copy())
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        fmin(f, self.search_space, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.RandomState(SEED))
        best_params = space_eval(self.search_space, trials.argmin)
        test_accuracy = self.eval_on_best(best_params)

class HyperoptMetaModelTest():
    '''
    Assumes a classification metamodel as it uses iris dataset for the test
    '''
    def __init__(self, meta_model):
        '''
        The  search space is expected to be a list of search spaces and this class with concatenate it to create a full search space for the pipeline
        '''
        self.meta_model = meta_model
        self.search_space = hp.choice('meta_model', [hyperopt_search_space(meta_model)])
        iris = sklearn.datasets.load_iris()
        X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target)
        self.X_train, self.y_train = X_all[5:], y_all[5:]
        self.X_test, self.y_test = X_all[:5], y_all[:5]

    def hyperopt_train_test(self, params):
        #Use the accuracy based on a random train-test split as the evaluation criterion
        X_train_part, X_validation, y_train_part, y_validation = train_test_split(self.X_train, self.y_train, test_size=0.20)

        clf = create_instance_from_hyperopt_search_space(self.meta_model, params)
        clf_trained = clf.fit(X_train_part, y_train_part)
        predictions = clf_trained.predict(X_validation)
        accuracy = accuracy_score(y_validation, [round(pred) for pred in predictions])
        return accuracy

    def eval_on_best(self, params):
        clf = create_instance_from_hyperopt_search_space(self.meta_model, params)
        clf_trained = clf.fit(self.X_train, self.y_train)
        predictions = clf_trained.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, [round(pred) for pred in predictions])
        return accuracy

    def test_meta_model(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        def f(params):
            acc = self.hyperopt_train_test(params)
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        fmin(f, self.search_space, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.RandomState(SEED))
        best_params = space_eval(self.search_space, trials.argmin)
        test_accuracy = self.eval_on_best(best_params)
        print("test_accuracy:{}".format(test_accuracy))

if __name__ == "__main__":
    from lale.lib.sklearn import LogisticRegression
    from lale.lib.sklearn import PCA
    tfm = PCA(n_components=3)
    clf = LogisticRegression(random_state=42)
    trainable = tfm >> clf
    clf2 = HyperoptMetaModelTest(trainable)
    clf2.test_meta_model()
