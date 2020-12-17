from pytorch_lightning.metrics import Metric
import torch
import numpy as np

from typing import Any, NoReturn, Optional
from pytorch_lightning.metrics.utils import METRIC_EPS, _input_format_classification_one_hot

class MyMetrics(Metric):
    def __init__(self, id2label ,dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.id2label=id2label
        self.label2id = {v:k for k,v in id2label.items()}

        self.pred_classes = []
        self.true_classes = []

        # self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("predicted", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        p, t = self.logits_and_one_hot_labels_to_string(logits=preds, one_hot_labels=target)	
        self.pred_classes.extend(p)	
        self.true_classes.extend(t)
        # OH_P = [[self.label2id[l] for l in single_batch] for single_batch in self.pred_classes]
        # OH_T = [[self.label2id[l] for l in single_batch] for single_batch in self.true_classes]
        # OH_CP = [[self.label2id[p] for p in s_batch_p if p in s_batch_t] for s_batch_p, s_batch_t in zip(self.pred_classes, self.true_classes)]

        # oh_p = torch.zeros(len(self.label2id))
        # oh_t = torch.zeros(len(self.label2id))
        # oh_cp = torch.zeros(len(self.label2id))

        # for sub in OH_P:
        #     for value in sub:
        #         oh_p[value] += 1
        # for sub in OH_T:
        #     for value in sub:
        #         oh_t[value] += 1
        # for sub in OH_CP:
        #     for value in sub:
        #         oh_cp[value] += 1
        

    def compute(self):
        assert len(self.pred_classes) == len(self.true_classes), "Error in id2label traduction"	

        avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1 = self.compute_metrics(pred_classes=self.pred_classes,	
                                                                                                true_classes=self.true_classes)
        
        self.pred_classes = []
        self.true_classes = []

        return avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1

    def logits_and_one_hot_labels_to_string(self, logits, one_hot_labels, no_void = False, threshold = 0.5):	

        pred_classes, true_classes = [], []	

        for example_logits, example_labels in zip(logits, one_hot_labels):	
            mask = example_logits > threshold	
            if no_void:	
                argmax = np.argmax(example_logits)	
                pc = self.id2label[argmax]	
                p_classes = [self.id2label[i] for i, m in enumerate(mask) if m]	

                if pc not in p_classes:	
                    p_classes.append(pc)	
                pred_classes.append(p_classes)	

            else:    	
                pred_classes.append([self.id2label[i] for i, m in enumerate(mask) if m])	
            true_classes.append([self.id2label[i] for i, l in enumerate(example_labels) if l])	

        assert len(pred_classes) == len(true_classes), "Error in id2label traduction"	
        return pred_classes, true_classes

    def compute_metrics(self, pred_classes, true_classes):	
        correct_counter = 0	
        prediction_counter = 0	
        true_labels_counter = 0	
        precision_sum = 0	
        recall_sum = 0	
        f1_sum = 0	

        void_prediction_counter = 0	

        for example_pred, example_true in zip(pred_classes, true_classes):	

            assert len(example_true) > 0, 'Error in true label traduction'	

            prediction_counter += len(example_pred)	

            true_labels_counter += len(example_true)	
            if not example_pred:	
                void_prediction_counter += 1	
            else:	
                correct_predictions = len(set(example_pred).intersection(set(example_true)))	
                correct_counter += correct_predictions	

                p = correct_predictions / len(example_pred)	
                r = correct_predictions / len(example_true)	
                f1 = self.compute_f1(p, r)	
                precision_sum += p	
                recall_sum += r	
                f1_sum += f1


        micro_p = correct_counter / prediction_counter	
        micro_r = correct_counter / true_labels_counter	
        micro_f1 = self.compute_f1(micro_p, micro_r)	

        examples_in_dataset = len(true_classes)	

        macro_p = precision_sum / examples_in_dataset	
        macro_r = recall_sum / examples_in_dataset	
        macro_f1 = f1_sum / examples_in_dataset	

        avg_pred_number = prediction_counter / examples_in_dataset	


        return avg_pred_number, void_prediction_counter, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

    def compute_f1(self, p, r):	
        return (2*p*r)/(p + r) if (p + r) else 0


class Precision(Metric):
    r"""
    Computes `Precision <https://en.wikipedia.org/wiki/Precision_and_recall>`_:

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively.  Works with binary, multiclass, and
    multilabel data.  Accepts logits from a model output or integer class
    values in prediction.  Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        beta: Beta coefficient in the F measure.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5

        average:
            * `'micro'` computes metric globally
            * `'macro'` computes metric for each class and then takes the mean

        multilabel: If predictions are from multilabel classification.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import Precision
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> precision = Precision(num_classes=3)
        >>> precision(preds, target)
        tensor(0.3333)

    """
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = 'micro',
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        assert self.average in ('micro', 'macro'), \
            "average passed to the function must be either `micro` or `macro`"

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        t_preds, t_target = _input_format_classification_one_hot(
            self.num_classes, preds, target, self.threshold, self.multilabel
        )
        batch_correct_preds = torch.sum(t_preds * t_target, dim=1)
        batch_all_preds = torch.sum(t_preds, dim=1)
        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += batch_correct_preds
        self.predicted_positives += batch_all_preds
        x = 1

    def compute(self):
        if self.average == 'micro':
            micro_p = self.true_positives.sum().float() / (self.predicted_positives.sum() + METRIC_EPS)
            return micro_p
        elif self.average == 'macro':
            macro_p = (self.true_positives.float() / (self.predicted_positives + METRIC_EPS)).mean()
            return (self.true_positives.float() / (self.predicted_positives + METRIC_EPS)).mean()


class Recall(Metric):
    r"""
    Computes `Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_:

    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively. Works with binary, multiclass, and
    multilabel data.  Accepts logits from a model output or integer class
    values in prediction.  Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        beta: Beta coefficient in the F measure.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5

        average:
            * `'micro'` computes metric globally
            * `'macro'` computes metric for each class and then takes the mean

        multilabel: If predictions are from multilabel classification.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import Recall
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> recall = Recall(num_classes=3)
        >>> recall(preds, target)
        tensor(0.3333)

    """
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = 'micro',
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        assert self.average in ('micro', 'macro'), \
            "average passed to the function must be either `micro` or `macro`"

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        t_preds, t_target = _input_format_classification_one_hot(
            self.num_classes, preds, target, self.threshold, self.multilabel
        )

        batch_correct_pred = torch.sum(t_preds * t_target, dim=1)
        batch_true_label = torch.sum(t_target, dim=1)
        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += batch_correct_pred
        self.actual_positives += batch_true_label
        x = 1

    def compute(self):
        """
        Computes accuracy over state.
        """
        if self.average == 'micro':
            micro_r = self.true_positives.sum().float() / (self.actual_positives.sum() + METRIC_EPS)
            return micro_r
        elif self.average == 'macro':
            macro_r =  (self.true_positives.float() / (self.actual_positives + METRIC_EPS)).mean()
            return macro_r
