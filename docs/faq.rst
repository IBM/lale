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
    pass them to automation tools to guide their search.

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
    implementation follows a scikit-learn API of fit/predict or
    fit/transform. You can cast the operator into a Lale operator
    using ``lale_op = lale.operators.make_operator(non_lale_op)``.  If
    you want to get full Lale support for your own operator, we have a
    separate guide for how to do that:
    https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_new_operators.ipynb

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
