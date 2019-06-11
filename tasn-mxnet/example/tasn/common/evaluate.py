import mxnet as mx

class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        self.num = num
        self.names =['att_net_accuracy', 'part_net_accuracy', 'master_net_accuracy', 'part_net_aux_accuracy', 'master_net_aux_accuracy', 'distillation_loss']
        super(Multi_Accuracy, self).__init__('multi-accuracy')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0 if self.num is None else [0] * (self.num)
        self.sum_metric = 0.0 if self.num is None else [0.0] * (self.num)

    def update(self, labels, preds):
       # mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num-1

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy()
            label = labels[i].asnumpy()
            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)
        pre = preds[-1].asnumpy()
        self.sum_metric[-1] += pre.sum()
        self.num_inst[-1] += len(labels[0].asnumpy().flat)


    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            return super(Multi_Accuracy, self).get()
        else:
            return zip(*((self.names[i], float('nan') if self.num_inst[i] == 0
                                                      else self.sum_metric[i] / self.num_inst[i])
                       for i in range(self.num)))

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self.num is None:
            return super(Multi_Accuracy, self).get_name_value()
        name, value = self.get()
        return list(zip(name, value))

