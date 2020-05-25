Frequently Asked Questions
==========================

- Can I work with other modalities besides tables?

  - Besides tables, we have successfully used Lale for text, images,
    and time-series. In fact, Lale even works for multi-modal data,
    using the ``&`` combinator to specify different preprocessing
    paths per modality.

- Can I work with other tasks besides classification?

  - Besides classification, we mostly use Lale for regression. That
    said, you can define your own scoring metrics for evaluation, and
    pass them to automation tools to guide their search. The following
    notebook includes an example ``disparate_impact_scorer``:    
    https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/demo_aif360.ipynb

- I get an error when I instantiate an operator imported from
  Lale. What's wrong?

  - Lale raises errors on invalid hyperparameter values or
    combinations. This ensures that the operators are used correctly.
    So don't be surprised if you get any errors when you initialize
    Lale operators with some hyperparameter values. Chances are that
    those hyperpameters or combinations of hyperparameters are
    invalid. If not, please contact us.

- The algorithm I want to use is not present in Lale. Can I still use
  it?

  - Some of the features of Lale can be used if the algorithm
    implementation follows the scikit-learn conventions of fit/predict or
    fit/transform. You can turn any such operator into a Lale operator
    using ``lale_op = lale.operators.make_operator(non_lale_op)``.  If
    you want to get full Lale support for your own operator, we have a
    separate guide for how to do that:
    https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_new_operators.ipynb

- Can I use Lale for deep learning?

  - There are multiple facets to this question. The Lale library
    already includes a few DL operators such as a BERT transformer,
    a ResNet classifier, and an MLP classifier. Lale can perform
    joint algorithm selection and hyperparameter optimization over
    pipelines involving these operators. Furthermore, users can wrap
    additional DL operators as described. On the other hand, Lale does
    not currently support full-fledged neural architecture
    search (NAS). It can only perform architecture search when that
    is exposed through hyperparameters, such as ``hidden_layer_sizes``
    for MLP.
    
- How does the search space generation work?

  - Lale includes a search space generator that takes in a planned
    pipeline and the schemas of the operators in that pipeline, and
    returns a search space for your auto-ML tool of choice. Our arXiv
    paper describes how that works in detail:
    https://arxiv.org/abs/1906.03957

- Does Lale optimize for computational performance?

  - While Lale focuses mostly on types and automation, we have also
    done a little bit of work on computational performance. However,
    it has not been a major focus. If you encounter pain-points,
    please reach out to us.

- What is the relationship between Lale and IBM products?

  - Lale is free and open-source and does not depend on any commercial
    products. It is available under the `Apache`_ license and only
    requires the other open-source packages listed in `setup.py`_.
    Lale is used by IBM's `AutoAI SDK`_. The AutoAI SDK provides API
    access to various services on IBM cloud, including advanced
    pipeline search optimizers, cloud-hosted notebooks in Watson
    Studio, storage for datasets in Cloud Object Storage, storage for
    historical pipelines, deploying trained pipelines as a scoring
    service in Watson Machine Learning, etc. Lale does not require
    the AutoAI SDK to run, you can use it without the AutoAI SDK.

    .. _`Apache`: https://github.com/IBM/lale/blob/master/LICENSE.txt
    .. _`setup.py`: https://github.com/IBM/lale/blob/master/setup.py
    .. _`AutoAI SDK`: https://dataplatform.cloud.ibm.com/exchange/public/entry/view/a2d87b957b60c846267137bfae130dca
