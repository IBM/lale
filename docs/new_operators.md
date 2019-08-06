
[//]: # (Do not edit the markdown version of this file directly, it is auto-generated from a notebook.)
# Wrapping New Individual Operators

Lale comes with several library operators, so you do not need to write
your own. But if you want to contribute new operators, this section is
for you.  First let us review some basic concepts in Lale from the
point of view of adding new operators (estimators and
transformers). LALE (Language for Automated Learning Exploration) is
designed for the following goals:

* Automation: facilitate automated search and composition of pipelines
* Portability: independent of library or programming language, cloud-ready
* Correctness: single source of truth, correct by construction, type-checked
* Usability: leverage sklearn mind-share, popularity, and codes

To enable the above properties for your operators with Lale, you need to:

1. Write an operator implementation class with methods `__init__`,
   `fit`, and `predict` or `transform`. If you have a custom estimator
   or transformer as per scikit-learn, you can skip this step as that
   is already a valid Lale operator.
2. Write a machine readable JSON schema to indicate what
   hyperparameters are expected by an operator, to specify the types,
   default values, and recommended minimum/maximum values for
   automatic tuning. The hyperparameter schema can also encode
   constraints indicating dependencies between hyperparameter values
   such as solver `abc` only supports penalty `xyz`.
3. Write JSON schemas to specify the inputs and outputs of fit and
   predict or transform. These are used for validating the data before
   using it.
4. Register the operator implementation with Lale by passing the
   operator implementation class (created in step 1) and a dictionary
   containing the schemas.
5. Test and use the new operator, for instance, for training or
   hyperparameter optimization.

The next sections illustrate these five steps using an example.  After
the example-driven sections, this document concludes with a reference
covering features from the example and beyond.  This document focuses
on individual operators. We will document pipelines that compose
multiple operators elsewhere.

## 1. Operator Implementation Class

This section can be skipped if you already have a scikit-learn
compatible estimator or transformer class with methods `__init__`,
`fit`, and `predict` or `transform`. Any other compatibility with
scikit-learn such as `get_params` or `set_params` is optional, and so
is extending from `sklearn.base.BaseEstimator`.

This section illustrates how to implement this class with the help of
an example. The running example in this document is a simple custom
operator that just wraps the `LogisticRegression` estimator from
scikit-learn. Of course you can write a similar class to wrap your own
operators, which do not need to come from scikit-learn.  The following
code defines a class `MyLRImpl` for it. We will later register this as
an operator with name `MyLR`.


```python
import sklearn.linear_model

class MyLRImpl:
    def __init__(self, solver='warn', penalty='l2', C=1.0):
        self._hyperparams = {
            'solver': solver,
            'penalty': penalty,
            'C': C }
        
    def fit(self, X, y):
        self._sklearn_model = sklearn.linear_model.LogisticRegression(
            **self._hyperparams)
        self._sklearn_model.fit(X, y)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
```

This code first imports the relevant scikit-learn package. Then, it declares
a new class for wrapping it. Currently, Lale only supports Python, but
eventually, it will also support other programming languages. Therefore, the
Lale approach for wrapping new operators carefully avoids depending too much
on the Python language or any particular Python library. Hence, the
`MyLRImpl` class does not need to inherit from anything, but it does need to
follow certain conventions:

* It has a constructor, `__init__`, whose arguments are the
  hyperparameters.

* It has a training method, `fit`, with an argument `X` containing the
  training examples and, in the case of supervised models, an argument `y`
  containing labels. The `fit` method creates an instance of the scikit-learn
  `LogisticRegression` operator, trains it, and returns the wrapper object.

* It has a prediction method, `predict` for an estimator or `transform` for
  a transformer. The method has an argument `X` containing the test examples
  and returns the labels for `predict` or the transformed data for
  `transform`.

These conventions are designed to be similar to those of scikit-learn.
However, they avoid a code dependency upon scikit-learn.

## 2. Writing the Schemas

Lale requires schemas both for error-checking and for generating search
spaces for hyperparameter optimization.
The schemas of a Lale operator specify the space of valid values for
hyperparameters, for the arguments to `fit` and `predict` or `transform`,
and for the output of `predict` or `transform`. To keep the schemas
independent of the Python programming language, they are expressed as
[JSON Schema](https://json-schema.org/understanding-json-schema/reference/).
JSON Schema is currently a draft standard and is already being widely
adopted and implemented, for instance, as part of specifying
[Swagger APIs](https://www.openapis.org/).

The first schema specifies the arguments of the `fit` method.


```python
_input_schema_fit = {
  '$schema': 'http://json-schema.org/draft-04/schema#',    
  'type': 'object',
  'required': ['X', 'y'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}},
    'y': {
      'type': 'array',
      'items': {'type': 'number', 'minimum': 0}}}}
```

A JSON schema is itself expressed as a JSON document, here represented using
Python syntax for dictionary and list literals. The `fit` method of `MyLR`
takes two arguments, `X` and `y`. The `X` argument is an array of arrays of
numbers. The outer array is over samples (rows) of a dataset. The inner
array is over features (columns) of a sample. The `y` argument is an array
of non-negative numbers. Each element of `y` is a label for the
corresponding sample in `X`.

The schema for the arguments of the `predict` method is similar, just
omitting `y`:


```python
_input_schema_predict = {
  '$schema': 'http://json-schema.org/draft-04/schema#',    
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}}}}
```

The output schema indicates that the `predict` method returns an array of
labels with the same schema as `y`:


```python
_output_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',    
  'type': 'array',
  'items': {'type': 'number', 'minimum': 0}}
```

The most sophisticated schema specifies hyperparameters. The running example
chooses hyperparameters of scikit-learn LogisticRegression that illustrate
all the interesting cases. More complete and elaborate examples can be found
in the Lale standard library. The following specifies each hyperparameter
one at a time, omitting cross-cutting constraints.


```python
_hyperparams_ranges = {
  '$schema': 'http://json-schema.org/draft-04/schema#',    
  'type': 'object',
  'additionalProperties': False,
  'required': ['solver', 'penalty', 'C'],
  'relevantToOptimizer': ['solver', 'penalty', 'C'],
  'properties': {
    'solver': {
      'description': 'Algorithm for optimization problem.',
      'enum': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
      'default': 'liblinear'},
    'penalty': {
      'description': 'Norm used in the penalization.',
      'enum': ['l1', 'l2'],
      'default': 'l2'},
    'C': {
      'description':
        'Inverse regularization strength. Smaller values specify '
        'stronger regularization.',
      'type': 'number',
      'distribution': 'loguniform',
      'minimum': 0.0,
      'exclusiveMinimum': True,
      'default': 1.0,
      'minimumForOptimizer': 0.03125,
      'maximumForOptimizer': 32768}}}
```

Here, `solver` and `penalty` are categorical hyperparameters and `C` is a
continuous hyperparameter. For all three hyperparameters, the schema
includes a description, used for interactive documentation, and a
default value, used when no explicit value is specified. The categorical
hyperparameters are then specified as enumerations of their legal values.
In contrast, the continuous hyperparameter is a number, and the schema
includes additional information such as its distribution, minimum, and
maximum. In the example, `C` has `'minimum': 0.0`, indicating that only
positive values are valid. Furthermore, `C` has a
`'minimumForOptimizer': 0.03125` and `'maxmumForOptimizer': 32768`,
guiding the optimizer to limit its search space.

Besides specifying hyperparameters one at a time, users may also want to
specify cross-cutting constraints to further restrict the hyperparameter
schema. This part is an advanced use case and can be skipped by novice
users.


```python
_hyperparams_constraints = {
  'allOf': [
    { 'description':
        'The newton-cg, sag, and lbfgs solvers support only l2 penalties.',
      'anyOf': [
        { 'type': 'object',
          'properties': {
            'solver': {'not': {'enum': ['newton-cg', 'sag', 'lbfgs']}}}},
        { 'type': 'object',
          'properties': {'penalty': {'enum': ['l2']}}}]}]}
```

In JSON schema, `allOf` is a logical "and", `anyOf` is a logical "or", and
`not` is a logical negation. Thus, the `anyOf` part of the example can be
read as

```python
assert not (solver in ['newton-cg', 'sag', 'lbfgs']) or penalty == 'l2'
```

By standard Boolean rules, this is equivalent to a logical implication:

```python
if solver in ['newton-cg', 'sag', 'lbfgs']:
    assert penalty == 'l2'
```

In this particular example, the top-level `allOf` only has a single
component and could thus be omitted. But in general, scikit-learn often
imposes several side constraints on operators, which should then be
connected via logical "and".

The complete hyperparameters schema simply combines the ranges with the
constraints:


```python
_hyperparams_schema = {
  'allOf': [_hyperparams_ranges, _hyperparams_constraints]}
```

Finally, combining all schemas together and adding tags for discovery
and documentation yields a comprehensive set of metadata for our new
`MyLR` operator. 


```python
_combined_schemas = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'type': 'object',
  'tags': {
    'pre': ['~categoricals'],
    'op': ['estimator', 'classifier', 'interpretable'],
    'post': ['probabilities']},
  'properties': {
    'input_fit': _input_schema_fit,
    'input_predict': _input_schema_predict,
    'output': _output_schema,
    'hyperparams': _hyperparams_schema } }
```

## 3. Testing and Using the new Operator

Once your operator implementation and schema definitions are ready,
you can test it with Lale as follows. First, you will need to install
Lale, as described in the
[installation](../../master/docs/installation.md)) instructions.

### 3.1. Test JSON Schemas

For debugging purposes, it is wise to check whether the schemas are
actually valid with respect to the JSON Schema standard. Before using
the operator with Lale, you can test the schemas as follows:


```python
import lale.helpers
lale.helpers.validate_is_schema(_combined_schemas)
```

### 3.2. Register the Operator with Lale

Lale offers several features such as automation (e.g., hyperparameter
optimization), interactive documentation (e.g., hyperparameter
descriptions), and validation (e.g., of hyperparameter values against
their schema). To take advantage of those, the following code creates
a Lale operator `MyLR` from the previously-defined Python class
`MyLRImpl` and JSON schemas `_combined_schemas`.


```python
import lale.operators
MyLR = lale.operators.make_operator(MyLRImpl, _combined_schemas)
```

### 3.3. Use the new Operator

Before demonstrating the new `MyLR` operator, the following code loads the
Iris dataset, which comes out-of-the-box with scikit-learn.


```python
import sklearn.datasets
import sklearn.utils
iris = sklearn.datasets.load_iris()
X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target, random_state=42)
holdout_size = 30
X_train, y_train = X_all[holdout_size:], y_all[holdout_size:]
X_test, y_test = X_all[:holdout_size], y_all[:holdout_size]
print('expected {}'.format(y_test))
```

    expected [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]


Now that the data is in place, the following code sets the hyperparameters,
calls `fit` to train, and calls `predict` to make predictions. This code
looks almost like what people would usually write with scikit-learn, except
that it uses an enumeration `MyLR.solver` that is implicitly defined by Lale
so users do not have to pass in error-prone strings for categorical
hyperparameters.


```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

trainable = MyLR(MyLR.solver.lbfgs, C=0.1)
trained = trainable.fit(X_train, y_train)
predictions = trained.predict(X_test)
print('actual {}'.format(predictions))
```

    actual [1 0 2 1 2 0 1 2 1 1 2 0 0 0 0 2 2 1 1 2 0 2 0 2 2 2 2 2 0 0]


To illustrate interactive documentation, the following code retrieves the
specification of the `C` hyperparameter.


```python
MyLR.hyperparam_schema('C')
```




    {'description': 'Inverse regularization strength. Smaller values specify stronger regularization.',
     'type': 'number',
     'distribution': 'loguniform',
     'minimum': 0.0,
     'exclusiveMinimum': True,
     'default': 1.0,
     'minimumForOptimizer': 0.03125,
     'maximumForOptimizer': 32768}



Similarly, operator tags are reflected via Python methods on the operator:


```python
print(MyLR.has_tag('interpretable'))
print(MyLR.get_tags())
```

    True
    {'pre': ['~categoricals'], 'op': ['estimator', 'classifier', 'interpretable'], 'post': ['probabilities']}


To illustrate error-checking, the following code showcases an invalid
hyperparameter caught by JSON schema validation.


```python
from lale.helpers import assert_raises_validation_error
with assert_raises_validation_error():
    MyLR(solver='adam')
```

    error:
      message: "Failed validating hyperparameters for MyLR due to 'adam' is not one of\
        \ ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']\n\nFailed validating 'enum'\
        \ in schema['allOf'][0]['properties']['solver']:\n    {'default': 'liblinear',\n\
        \     'description': 'Algorithm for optimization problem.',\n     'enum': ['newton-cg',\
        \ 'lbfgs', 'liblinear', 'sag', 'saga']}\n\nOn instance['solver']:\n    'adam'"
      schema: !!python/object:jsonschema._utils.Unset {}


Finally, to illustrate hyperparameter optimization, the following code uses
[hyperopt](http://hyperopt.github.io/hyperopt/). We will document the
hyperparameter optimization use-case in more detail elsewhere. Here we only
demonstrate that Lale with `MyLR` supports it. 


```python
from lale.search.op2hp import hyperopt_search_space
from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def objective(hyperparams):
    del hyperparams['name']
    trainable = MyLR(**hyperparams)
    trained = trainable.fit(X_train, y_train)
    predictions = trained.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {'loss': -accuracy, 'status': STATUS_OK}

#The following line is enabled by the hyperparameter schema.
search_space = hyperopt_search_space(MyLR)

trials = Trials()
fmin(objective, search_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_hyperparams = space_eval(search_space, trials.argmin)
print('best hyperparameter combination {}'.format(best_hyperparams))
```

    100%|█████████████████████████| 10/10 [00:00<00:00, 24.47it/s, best loss: -1.0]
    best hyperparameter combination {'C': 12866.556345415156, 'name': '__main__.MyLR', 'penalty': 'l2', 'solver': 'liblinear'}


This concludes the running example. To
summarize, we have learned how to write an operator implementation class and JSON
schemas; how to register the Lale operator; and how to use the Lale operator for
manual as well as automated machine-learning.

## 4. Additional Wrapper Class Features

Besides `X` and `y`, the `fit` method in scikit-learn sometimes has
additional arguments. Lale also supports such additional arguments.

In addition to the `__init__`, `fit`, and `predict` methods, many
scikit-learn estimators also have a `predict_proba` method. Lale will
support that with its own metadata schema.

## 5. Reference

This section documents features of JSON Schema that Lale uses, as well as
extensions that Lale adds to JSON schema for information specific to the
machine-learning domain. For a more comprehensive introduction to JSON
Schema, refer to its
[Reference](https://json-schema.org/understanding-json-schema/reference/).

The following table lists kinds of schemas in JSON Schema:

| Kind of schema | Corresponding type in Python/Lale |
| ---------------| ---------------------------- |
| `null`         | `NoneType`, value `None` |
| `boolean`      | `bool`, values `True` or `False` |
| `string`       | `str` |
| `enum`         | See discussion below. |
| `number`       | `float`, .e.g, `0.1` |
| `integer`      | `int`, e.g., `42` |
| `array`        | See discussion below. |
| `object`       | `dict` with string keys |
| `anyOf`, `allOf`, `not` | See discussion below. |

The use of `null`, `boolean`, and `string` is fairly straightforward.  The
following paragraphs discuss the other kinds of schemas one by one.

### 5.1. enum

In JSON Schema, an enum can contain assorted values including strings,
numbers, or even `null`. Lale uses enums of strings for categorical
hyperparameters, such as `'penalty': {'enum': ['l1', 'l2']}` in the earlier
example. In that case, Lale also automatically declares a corresponding
Python `enum`.
When Lale uses enums of other types, it is usually to restrict a
hyperparameter to a single value, such as `'enum': [None]`.

### 5.2. number, integer

In schemas with `type` set to `number` or `integer`, JSON schema lets users
specify `minimum`, `maximum`,
`exclusiveMinimum`, and `exclusiveMaximum`. Lale further extends JSON schema
with `minimumForOptimizer`, `maximumForOptimizer`, and `distribution`.
Possible values for the `distribution` are `'uniform'` (the default) and
`'loguniform'`. In the case of integers, Lale quantizes the distributions
accordingly.

### 5.3. array

Lale schemas for input and output data make heavy use of the JSON Schema
`array` type. In this case, Lale schemas are intended to capture logical
schemas, not physical representations, similarly to how relational databases
hide physical representations behind a well-formalized abstraction layer.
Therefore, Lale uses arrays from JSON Schema for several types in Python.
The most obvious one is a Python `list`. Another common one is a numpy
[ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html),
where Lale uses nested arrays to represent each of the dimensions of a
multi-dimensional array. Lale also has support for `pandas.DataFrame` and
`pandas.Series`, for which it again uses JSON Schema arrays.

For arrays, JSON schema lets users specify `items`, `minItems`, and
`maxItems`. Lale further extends JSON schema with `minItemsForOptimizer` and
`maxItemsForOptimizer`. Furthermore, Lale supports a `typeForOptimizer`,
which can be `'tuple'` to support cases where the Python code requires a
tuple instead of a list.

### 5.4. object

For objects, JSON schema lets users specify a list `required` of properties
that must be present, a dictionary `properties` of sub-schemas, and a flag
`additionalProperties` to indicate whether the object can have additional
properties beyond the keys of the `properties` dictionary. Lale further
extends JSON schema with a `relevantToOptimizer` list of properties that
hyperparameter optimizers should search over.

For individual properties, Lale supports a `default`, which is inspired by
and consistent with web API specification practice. It also supports a
`forOptimizer` flag which defaults to `True` but can be set to `False` to
hide a particular subschema from the hyperparameter optimizer. For example,
the number of components for PCA in scikit-learn can be specified as an
integer or a floating point number, but an optimizer should only explore one
of these choices.

### 5.5. allOf, anyOf, not

As discussed before, in JSON schema, `allOf` is a logical "and", `anyOf` is
a logical "or", and `not` is a logical negation. The running example from
earlier already illustrated how to use these for implementing cross-cutting
constraints. Another use-case that takes advantage of `anyOf` is for
expressing union types, which arise frequently in scikit-learn. For example,
here is the schema for `n_components` from PCA:

```python
'n_components': {
  'anyOf': [
    { 'description': 'If not set, keep all components.',
      'enum': [None]},
    { 'description': "Use Minka's MLE to guess the dimension.",
      'enum': ['mle']},
    { 'description':
        'Select the number of components such that the amount of variance '
        'that needs to be explained is greater than the specified percentage.',
      'type': 'number',
      'minimum': 0.0, 'exclusiveMinimum': True,
      'maximum': 1.0, 'exclusiveMaximum': True},
    { 'description': 'Number of components to keep.',
      'type': 'integer',
      'minimum': 1,
      'forOptimizer': False}],
  'default': None}
```

### 5.6. Schema Metadata

We encourage users to make their schemas more readable by also including
common JSON schema metadata such as `$schema` and `description`.  As seen in
the examples in this document, Lale also extends JSON schema with `tags`
and `documentation_url`. Finally, in some cases, schema-internal
duplication can be avoided by cross-references and linking. This is
supported by off-the-shelf features of JSON schema without requiring
Lale-specific extensions.
