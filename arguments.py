import argparse

# coco: /root/pan/netDiskS/zy/dataset/coco
# f30k: /root/pan/netDiskS/zy/dataset/f30k
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/root/pan/netDiskS/zy/dataset/coco',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--optim', default='adam', type=str,
                        help='the optimizer')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|backbone')


    # add input by zy
    
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--use_moco', default=1, type=int)

    parser.add_argument('--moco_M', default=4096, type=int)

    parser.add_argument('--moco_r', default=0.999, type=float)

    parser.add_argument('--loss_lamda', default=1, type=int) 
    
    parser.add_argument('--mu', default=90, type=float)  

    parser.add_argument('--gama', default=0.5, type=float)    



    ## new
    parser.add_argument('--n_layer', default=3, type=int)

    parser.add_argument('--d_k', default=128, type=int)

    parser.add_argument('--d_v', default=128, type=int)

    parser.add_argument('--head', default=8, type=int)

    parser.add_argument('--d_model', default=1024, type=int)

    parser.add_argument('--train_mode', default='r2g', help='r2g, g2r')

    parser.add_argument('--use_mod', default=0, type=int)

    parser.add_argument('--mod_alpha', default='0.2', type=float)

    parser.add_argument('--use_predict', default=1, type=int)
    
    parser.add_argument('--gpo_step', default=32, type=int)
    
    parser.add_argument('--use_angle_loss', default=1, type=int)
    
    parser.add_argument('--angle_loss_ratio', default=0.1, type=float)


    return parser
