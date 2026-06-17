import warnings

warnings.filterwarnings("ignore")
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到 Python 路径
sys.path.insert(0, current_dir)

# 然后导入其他模块
from models.trainer import *
from argparse import ArgumentParser
import torch
from models.trainer import *

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    print(f"Batch size: {args.batch_size}")
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print(f"Using device: {device}")
    dataloaders = utils.get_loaders(args)
    print(f"Training data loaded with {len(dataloaders['train'].dataset)} samples.")
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    print(f"Test data loaded with {len(dataloader.dataset)} samples.")
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='lucannet6_LEVIR_n3', type=str)
    parser.add_argument('--checkpoint_root', default='E:\data\STADE-CDNet\LEVIR', type=str)
    parser.add_argument('--vis_root', default='./visualizations', type=str)
    parser.add_argument('--output_folder', default='./output_results/predict_STADE-CD', type=str)
    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default="8", type=int, help='The parameters I set = 8')  #
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[1.0, 1.0],
                        help="Multi-scale prediction weights, e.g., [1.0, 0.5]")

    parser.add_argument('--net_G', default='lucannet', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default="0.00009567", type=float, help='The parameters I set = 0.00009567')
    parser.add_argument('--max_epochs', default=189, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
