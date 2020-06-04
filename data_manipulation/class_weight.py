import numpy as np
from sklearn.utils import class_weight

def get_class_weights(y_train):
        labels = np.argmax(y_train, axis=-1).flat

        cls_weight = class_weight.compute_class_weight(
                'balanced',
                np.unique(labels),
                labels
              )
        #quadratic_weights = cls_weight * cls_weight
        return cls_weight