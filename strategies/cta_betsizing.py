import pandas as pd
import pandas as pd
import sklearn


class SingleAssetStrategy:
    '''
    This is for cta
    TODO: adopt the 'next' architecture from backtesting.py
    '''

    def __init__(self, params: dict):
        self.parameters = params

    def run(self, features: pd.DataFrame, fitted_model: sklearn.base.BaseEstimator) -> pd.Series:
        '''
        converts a prediction into a delta (eg take position above proba_threshold proba, up to max_leverage for 100% proba)
        '''
        if isinstance(fitted_model, sklearn.base.ClassifierMixin):
            probas = getattr(fitted_model, 'predict_proba')(features)

            # prob->delta
            if self.parameters['type'] == 'threshold_leverage':
                expectation_threshold = self.parameters['params']['expectation_threshold']
                max_leverage = self.parameters['params']['max_leverage']

                threshold_applied = np.sign(fitted_model.classes_) * np.clip(np.abs(fitted_model.classes_) - expectation_threshold, a_min=0, a_max=999999999)
                leverage_applied = np.sign(threshold_applied) * max_leverage * np.clip(abs(threshold_applied), a_min=0, a_max=1)
                deltas = np.dot(leverage_applied, probas.transpose())
            elif self.parameters['type'] == 'max_loss_probability':
                max_loss_probability = self.parameters['params']['max_loss_probability']

                deltas = []
                class_down = np.array([1 if 'down' in class_ else 0 for class_ in fitted_model.classes_])
                class_up = np.array([1 if 'up' in class_ else 0 for class_ in fitted_model.classes_])
                class_deltas = np.array([self.parameters['params']['class_deltas'][class_] for class_ in fitted_model.classes_])
                for p in probas:
                    down_proba = sum(p * class_down)
                    up_proba = sum(p * class_up)
                    if (down_proba < max_loss_probability and up_proba > max_loss_probability) \
                            or (up_proba < max_loss_probability and down_proba > max_loss_probability):
                        deltas.append(sum(p * class_deltas))
                    else:
                        deltas.append(0.0)
            else:
                raise NotImplementedError
        elif isinstance(fitted_model, sklearn.base.RegressorMixin):
            deltas = getattr(fitted_model, 'predict')(features)
        else:
            raise NotImplementedError

        result = pd.Series(index=features.index, data=deltas)
        return result