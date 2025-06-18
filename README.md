# HiCer
Hierarchical and complementary experts transformer with momentum invariance for image-text retrieval, KBS，2025

Our source code of HiCer accepted by KBS, whihc is built on top of the [vse_inf](https://github.com/woodfrog/vse_infty), [USER](https://github.com/zhangy0822/USER), and [DLCT](https://github.com/luo3300612/image-captioning-DLCT) in PyTorch. 


## Training
Train MSCOCO and Flickr30K from scratch:

Modify the corresponding arguments and run `train_coco.sh` or `train_f30k.sh`

## Evaluation
Modify the corresponding arguments in `eval.py` and run `python eval.py`, `python eval_ensemble.py` for the final results.

## Please use the following bib entry to cite this paper if you are using any resources from the repo.
```
@article{zhang2025hierarchical,
  title={Hierarchical and complementary experts transformer with momentum invariance for image-text retrieval},
  author={Zhang, Yan and Ji, Zhong and Pang, Yanwei and Han, Jungong},
  journal={Knowledge-Based Systems},
  volume={309},
  pages={112912},
  year={2025},
  publisher={Elsevier}
}
```
