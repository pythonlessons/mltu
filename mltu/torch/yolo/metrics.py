from mltu.torch.metrics import Metric

import torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.metrics import Metric as YoloMetric
from ultralytics.utils.metrics import ConfusionMatrix, box_iou


class YoloMetrics(Metric):
    """ Accuracy metric class
    
    Args:
        name (str, optional): name of metric. Defaults to 'YoloMetrics'.
    """
    def __init__(self, nc, name: str="YoloMetrics", conf:float=0.001, iou:float=0.7, max_det:int=300) -> None:
        super(YoloMetrics, self).__init__(name=name)
        self.nc = nc
        self.jdict = []
        self.confusion_matrix = ConfusionMatrix(nc=nc, conf=conf)
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

        self.conf = conf
        self.iou = iou
        self.lb = []
        self.multi_label=True
        self.single_cls=False
        self.max_det = max_det
        self.seen = 0
        self.device = torch.device("cpu")
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.box = YoloMetric()
        self.loss_info = {"box_loss": [], "cls_loss": [], "dfl_loss": []}

    @property
    def keys(self):
        return ["precision", "recall", "mAP50", "mAP50-95", "fitness"]

    def reset(self):
        self.__init__(self.nc, self.name)

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1).to(self.device)
        bbox = batch["bboxes"][idx].to(self.device)
        ori_shape = batch["ori_shape"][si]
        # imgsz = batch["img"].shape[2:]
        imgsz = batch["resized_shape"][si]
        ratio_pad = (batch["resized_shape"][si][0] / batch["ori_shape"][si][0], batch["resized_shape"][si][1] / batch["ori_shape"][si][1])
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def update_stats(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue

            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = pred.clone().to(self.device)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            labels=self.lb,
            multi_label=self.multi_label,
            agnostic=self.single_cls,
            max_det=self.max_det,
        )

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        """ Update metric state with new data

        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target of data
        """
        # outputs = np.squeeze(output)
        model = kwargs.get("model")
        loss_info = kwargs.get("loss_info", None)
        if loss_info:
            for k, v in loss_info[0].items():
                self.loss_info[k].append(v.cpu().numpy())

        if not model.training: # update metrics only in validation mode
            preds = self.postprocess(output[0])
            self.update_stats(preds, target)

    def result(self) -> dict:
        """ Return metric value"""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items() if v}
        loss_info_stats = {k: np.mean(v) for k, v in self.loss_info.items()}
        if not stats:
            if loss_info_stats:
                return loss_info_stats
            return None
        
        # Compute statistics
        ap_per_class_results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=False,
            names={},
        )[2:]
        self.box.nc = self.nc
        self.box.update(ap_per_class_results)

        results = dict(zip(self.keys, self.box.mean_results() + [self.box.fitness()]))
        if loss_info_stats:
            results.update(loss_info_stats)

        return results