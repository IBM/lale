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

from lale.search.op2hp import hyperopt_search_space
from lale.lib.sklearn import LogisticRegression
from sklearn.datasets import load_iris
import sklearn.utils
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 3141

iris = sklearn.datasets.load_iris()
X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target)
X_train, y_train = X_all[5:], y_all[5:]
X_test, y_test = X_all[:5], y_all[:5]

def hyperopt_train_test(params):
    t = params['name']
    del params['name']
    clf = get_classifier(t, params)
    clf_trained = clf.fit(X_train, y_train)
    predictions = clf_trained.predict(X_test)
    accuracy = accuracy_score(y_test, [round(pred) for pred in predictions])
    return accuracy

def eval_on_best(params):
    print('Best algo and parameters:', params)
    t = params['name']
    del params['name']

    clf = get_classifier(t, params)
    clf_trained = clf.fit(X_train, y_train)
    predictions = clf_trained.predict(X_test)
    accuracy = accuracy_score(y_test, [round(pred) for pred in predictions])
    return accuracy

def get_classifier(t, param_dict):
    if 'LogisticRegression' in t:
        clf = LogisticRegression(**param_dict)
    else:
        return 0
    return clf

search_space = hp.choice('classifier', [hyperopt_search_space(LogisticRegression)])

count = 0
best = 0
def f(params):
    global best, count
    count += 1
    acc = hyperopt_train_test(params.copy())
    if acc > best:
        print('new best:', acc, 'using', params['name'])
        best = acc
    if count % 1 == 0:
        print('iters:', count, ', acc:', acc, 'using', params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
fmin(f, search_space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=np.random.RandomState(SEED))
best_params = space_eval(search_space, trials.argmin)
print('best:', best_params)
test_accuracy = eval_on_best(best_params)
print('Test Accuracy:', test_accuracy)




