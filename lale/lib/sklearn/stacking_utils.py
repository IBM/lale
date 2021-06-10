import pandas as pd
from sklearn.ensemble._stacking import _BaseStacking


class _BaseStackingLale(_BaseStacking):
    def _concatenate_predictions_pandas(self, X, predictions):
        X_meta = []
        idx = 0
        for est_idx, preds in enumerate(predictions):
            # case where the the estimator returned a 1D array
            if preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
            else:
                if (
                    self.stack_method_[est_idx] == "predict_proba"
                    and len(self.classes_) == 2
                ):
                    # Remove the first column when using probabilities in
                    # binary classification because both features are perfectly
                    # collinear.
                    X_meta.append(preds[:, 1:])
                else:
                    X_meta.append(preds)
            X_meta[-1] = pd.DataFrame(
                X_meta[-1],
                columns=[
                    f"estimator_{idx}_feature_{i}" for i in range(X_meta[-1].shape[1])
                ],
            )
            idx += 1
        if self.passthrough:
            X_meta.append(X)

        return pd.concat(X_meta, axis=1)
