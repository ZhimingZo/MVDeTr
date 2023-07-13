# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops.boxes import box_area
import torch.distributed as dist


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost (removed)
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        #print(bs, num_queries, outputs["pred_logits"].shape) # torch.Size([1, 100, 2])
        #print(bs, num_queries, outputs["pred_boxes"].shape) #  torch.Size([1, 100, 2])
        #print(targets)
        #print(targets[0]["labels"].shape) #torch.Size([1, 21]
        #print(targets[0]["boxes"].shape) # torch.Size([1, 21, 2])
        #exit()
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]  #[100, 2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4] #[100, 2]
        
        
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        #print(tgt_ids)
        #print(out_prob.shape)
        #exit()
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        #print(tgt_ids.shape, type(tgt_ids))
        #print(tgt_bbox.shape, type(tgt_bbox))
        #exit()
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


        '''
        print(out_bbox.shape, tgt_bbox.shape)
        if torch.isnan(out_bbox).any() or torch.isinf(out_bbox).any():
            print("Matrix contains invalid numeric entries.")
            matrix = out_bbox
            print("out_box")
            invalid_indices = torch.nonzero(torch.isnan(matrix) | torch.isinf(matrix), as_tuple=False)
            for i, j in invalid_indices:
                print(f"Invalid value at index ({i}, {j}): {matrix[i, j]}")
            exit()
        if torch.isnan(tgt_bbox).any() or torch.isinf(tgt_bbox.any()):
            print("Matrix contains invalid numeric entries.")
            matrix = tgt_bbox
             
            invalid_indices = torch.nonzero(torch.isnan(matrix) | torch.isinf(matrix), as_tuple=False)
            for i, j in invalid_indices:
                print(f"Invalid value at index ({i}, {j}): {matrix[i, j]}")
            exit()
        '''
        # Compute the giou cost betwen boxes
        #cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class # + self.cost_giou * cost_giou
        #C = cost_bbox + cost_class
        #C = self.cost_class * cost_class
        #C = self.cost_bbox * cost_bbo

        C = C.view(bs, num_queries, -1).cpu()
        '''
        if torch.isnan(C[0]).any() or torch.isinf(C[0]).any():
            print("Matrix contains invalid numeric entries.")
            matrix = C[0]
            print(matrix[20, 0])
            invalid_indices = torch.nonzero(torch.isnan(matrix) | torch.isinf(matrix), as_tuple=False)
            for i, j in invalid_indices:
                print(f"Invalid value at index ({i}, {j}): {matrix[i, j]}")
            exit()
        '''
        sizes = [len(v["boxes"]) for v in targets] # 1
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher_ce(args):
    if args is None: 
        return HungarianMatcher()
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox) #, cost_giou=args.set_cost_giou)
    #return HungarianMatcher()