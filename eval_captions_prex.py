from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap


eval_gt_coco = COCO('./val/captions_val2014.json')
eval_result_coco = eval_gt_coco.loadRes('./val/results.json')
scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
scorer.evaluate()
print('complete')
