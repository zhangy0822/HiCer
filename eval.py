import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='f30k',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/home/zy/dev/datasets/f30k')
    parser.add_argument('--trained_time', default='butd_region_bert_0516_151633')
    parser.add_argument('--save_results', action='store_true', default=True)
    # parser.add_argument('--evaluate_cxc', action='store_true',default=True)
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()
    
    if opt.dataset == 'coco':
        weights_bases = [
            './runs/coco',
            # './runs/release_weights/coco_butd_grid_bert',
            # './runs/release_weights/coco_wsl_grid_bert',
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            './runs/f30k',
            # './runs/f30k_butd_grid_bert',
            # './runs/f30k_wsl_grid_bert',
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        model_path = os.path.join(base, opt.trained_time, 'checkpoints/model_best.pth')
        logger.info('Evaluating {}...'.format(model_path))
        if opt.save_results and not opt.evaluate_cxc:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, opt.trained_time, 'results_{}.npy'.format(opt.dataset))
        elif opt.save_results and opt.evaluate_cxc:
            save_path = os.path.join(base, opt.trained_time, 'cxc_results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True, save_path=save_path)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    main()
