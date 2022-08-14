# KDD 2022 Hands-on Tutorial on "Gradual AutoML using Lale"

Tuesday 16 August, 1:30pm-4:30pm, Room 102-b

Authors: Kiran Kate, Martin Hirzel, Parikshit Ram, Avraham Shinnar,
and Jason Tsay (IBM Research)

https://kdd.org/kdd2022/handsOnTutorial.html

### Abstract

Lale is a sklearn-compatible library for automated machine learning
(AutoML).
It is open-source (https://github.com/ibm/lale) and
addresses the need for gradual automation of machine learning
as opposed to offering a black-box AutoML tool.
Black-box AutoML tools are difficult to customize and thus restrict
data scientists in leveraging their knowledge and intuition in the
automation process.
Lale is built on three principles: progressive disclosure,
orthogonality, and least surprise.
These enable a gradual approach offering a spectrum of usage patterns
starting from total automation to controlling almost every aspect of
AutoML.
Lale provides compositional constructs that let data scientists
control some aspects of their pipelines while leaving other aspects
free to be searched automatically.
This tutorial demonstrates the use of Lale for various
machine-learning tasks, showing how to progressively exercise more
customization.
It also covers AutoML for advanced scenarios such as class imbalance correction,
bias detection and mitigation, multi-objective optimization, and
working with multi-table datasets.
While Lale comes with hyperparameter specifications for 216
operators out-of-the-box, users can also add more operators of their
own, and this tutorial covers how to do that.
Overall, this tutorial teaches you how you can exercise fine-grained
control over AutoML without having to be an AutoML expert.

### Target Audience and Prerequisites

This tutorial targets data scientists who want to leverage AutoML.
It expects some familiarity with Python libraries such as numpy,
pandas, and sklearn
(our [user study](https://proceedings.neurips.cc/paper/2021/file/a3b36cb25e2e0b93b5f334ffb4e4064e-Paper.pdf)
showed that data scientists with moderate sklearn experience can
successfully use Lale to solve advanced tasks).
No AutoML experience is required.

### Tutorial Outline

#### 1. Introduction to AutoML

[01_intro.ipynb](01_intro.ipynb)

This notebook will provide an overview of AutoML in general and
introduce concepts such as hyperparameter optimization (HPO) and
combined algorithm selection and hyperparameter tuning (CASH).
We will also explain the concept of gradual AutoML.

#### 2. Total Automation with Lale

[02_total.ipynb](02_total.ipynb)

Lale provides a simple sklearn-style operator called `AutoPipeline`
to achieve total automation for standard machine learning tasks such as
classification and regression on tabular data.
We will demonstrate how to use `AutoPipeline` in just 3 lines of code,
and how to inspect the output models of `AutoPipeline`.

#### 3. Customizing Algorithm Choices and Hyperparameters

[03_custom.ipynb](03_custom.ipynb)

This notebook will focus on using Lale to compose a pipeline
with algorithm choices, such as between different categorical
encoders and between different classifiers.
Users can then perform CASH on this pipeline using
`auto_configure`.
We will also demonstrate how to refine outputs of AutoML and do
iterative AutoML.

#### 4. Handling Class Imbalance

[04_imbalance.ipynb](04_imbalance.ipynb)

Lale includes operators for class imbalance correction from
[imbalanced-learn](https://imbalanced-learn.org).
This notebook introduces the concept of higher-order operators and
shows how to use them to create pipelines involving down-sampling and
up-sampling.

#### 5. Bias Mitigation

[05_bias.ipynb](05_bias.ipynb)

To the best of our knowledge, Lale is the first AutoML library
that allows easy inclusion of bias mitigators in the search space.
This notebook introduces fairness metrics and bias mitigators from
[AIF360](https://aif360.mybluemix.net/) and demonstrates CASH over
them.

#### 6. Multi-objective Optimization

[06_multobj.ipynb](06_multobj.ipynb)

Most commonly used AutoML tools optimize for a single metric.
In practice, there is often a need to take more than one metric into
account while searching for the best model.
For example, we may want good
predictive performance as well as model fairness.
This notebook will demonstrate the use of Lale for such multi-objective
optimization to find Pareto frontiers.

#### 7. Working with Multi-table Datasets

[07_multtab.ipynb](07_multitab.ipynb)

The preprocessing steps in a data science workflow often involve more
than one table.
Data scientists usually write independent scripts for such data
preparation as there is no easy way to include it in a machine
learning pipeline.
Lale introduced preprocessing operators for performing join, filter,
map, groupby, aggregate, etc. as part of end-to-end sklearn-style
pipelines.
We will demonstrate these operators in the context of a standard
machine learning task.

#### 8. Adding a New Operator

[08_newops.ipynb](08_newops.ipynb)

An operator in Lale is a light-weight wrapper over existing algorithmic
implementations.
For example, Lale has wrappers for 216 existing implementations from
sklearn, imbalanced-learn, AIF360, etc.
Adding a new operator in Lale is relatively straightforward and well
documented.
This notebook walks through the steps to add a new operator wrapper and
to define a search space for that operator's hyperparameters.

#### 9. Scikit-learn compatibility

[09_compat.ipynb](09_compat.ipynb)

This notebook will discuss how lale provides compatibility with
scikit-learn based code.

#### 10. Uses for Schemas 

[10_schemas.ipynb](10_schemas.ipynb)

This notebook will discuss how Lale takes advantage of schemas, and
employs compiler techniques to generate search spaces for different
optimizer backends, for early error reporting, and for automatic
documentation generation.

#### 11. Research Directions

[11_directions.ipynb](11_directions.ipynb)

We will wrap up the tutorial with examples of some research directions:
(1) batch-wise training of pipelines for datasets that do not fit in
main memory, and
(2) grammars to define recursive search spaces.
We will cover examples of when and how these capabilities can be used.

### Acknowledgements

We would like to thank
Guillaume Baudart (DI ENS, Ecole normale superieure, PSL University, CNRS, INRIA, France),
Michael Feffer (Carnegie Mellon University, USA),
Louis Mandel (IBM Research, USA),
Chirag Sahni (Rensselaer Polytechnic Institute, USA), and
Vaibhav Saxena (IBM Research, India)
for their valuable contributions to Lale which will be included in the
tutorial.

### To Cite

```
@InProceedings{kate_et_al_2022,
  author = "Kate, Kiran and Hirzel, Martin and Ram, Parikshit and Shinnar, Avraham and Tsay, Jason",
  title = "Gradual {AutoML} using {Lale}",
  booktitle = "Tutorial at the Conference on Knowledge Discovery and Data Mining (KDD-Tutorial)",
  year = 2022,
  month = aug,
  url = "https://doi.org/10.1145/3534678.3542630" }
```
