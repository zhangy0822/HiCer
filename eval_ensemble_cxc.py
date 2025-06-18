import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Evaluate model ensemble
paths = ['runs/coco/butd_region_bert_0302_203736/cxc_results_coco.npy',
         'runs/coco/butd_region_bert_0303_001143/cxc_results_coco.npy']

evaluation.eval_ensemble_cxc(results_paths=paths)
