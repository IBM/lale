class _FitSpecProxy:
    def __init__(self, base):
        self._base = base

    def __getattr__(self, item):
        return getattr(self._base, item)

    def get_params(self, deep=True):
        ret = {}
        ret["base"] = self._base
        return ret

    def fit(self, X, y, sample_weight=None, **fit_params):
        return self._base.fit(X, y, sample_weight=sample_weight, **fit_params)
