import os
import argparse
import itertools
import torch
import random
import numpy as np
from model import MvCAN
from util import get_logger
from datasets import *
from configure import get_default_config
from weak_quality import apply_weak_quality_protocol, save_corruption_audit
import warnings
warnings.filterwarnings("ignore")


def set_global_seed(seed):
    seed = int(seed)

    # This records the intended hash seed for child processes. Python hash
    # initialization for this process has already happened before this call.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dataset = {
    1: "BDGP",
    2: "NoisyBDGP",
    3: "DIGIT",
    4: "NoisyDIGIT",
    5: "COIL",
    6: "NoisyCOIL",
    7: "Amazon",
    8: "NoisyAmazon",
    9: "DHA",
    10: "RGB-D",
    11: "Caltech-6V",
    12: "YoutubeVideo",
    13: "MSRC-v1",
}
Test = False
# Test = True
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='1', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--seed_override', type=int, default=None)
parser.add_argument('--init_epoch_override', type=int, default=None)
parser.add_argument('--epoch_override', type=int, default=None)
parser.add_argument('--batch_size_override', type=int, default=None)
parser.add_argument('--t1_override', type=int, default=None)
parser.add_argument('--t2_override', type=int, default=None)
parser.add_argument('--lr_override', type=float, default=None)
parser.add_argument('--lambda1_override', type=float, default=None)
parser.add_argument('--save_dir', type=str, default='./models')
parser.add_argument(
    '--corruption',
    choices=('none', 'heterogeneous_gaussian'),
    default='none',
)
parser.add_argument('--corruption_k', type=int, default=2)
parser.add_argument('--snr_db', type=float, default=5.0)
parser.add_argument('--corruption_seed', type=int, default=None)
parser.add_argument('--corruption_audit_dir', type=str, default=None)
args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    accs = []
    nmis = []
    aris = []
    # Configure
    config = get_default_config(dataset)
    training_overrides = {
        'seed': args.seed_override,
        'init_epoch': args.init_epoch_override,
        'epoch': args.epoch_override,
        'batch_size': args.batch_size_override,
        'T_1': args.t1_override,
        'T_2': args.t2_override,
        'lr': args.lr_override,
        'lambda1': args.lambda1_override,
    }
    for key, value in training_overrides.items():
        if value is not None:
            config['training'][key] = value
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    seed = int(config['training']['seed'])
    set_global_seed(seed)

    logger = get_logger()
    print("Dataset: " + config['dataset'])
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    print("Reproducibility seed: " + str(seed))
    print("Python random seeded: True")
    print("NumPy random seeded: True")
    print("Torch random seeded: True")
    print("KMeans random_state: " + str(seed))
    print("cuDNN deterministic: " + str(torch.backends.cudnn.deterministic))
    print("cuDNN benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # Load data
    X_list, Y_list = load_data(config)
    corruption_seed = args.corruption_seed
    if args.corruption == 'heterogeneous_gaussian' and corruption_seed is None:
        corruption_seed = seed
    X_list, corruption_audit = apply_weak_quality_protocol(
        X_list=X_list,
        mode=args.corruption,
        k=args.corruption_k,
        snr_db=args.snr_db,
        corruption_seed=corruption_seed,
    )
    n_clusters = len(np.unique(Y_list[0]))
    corruption_audit.update({
        'dataset': config['dataset'],
        'n_clusters': int(n_clusters),
        'model_seed': seed,
    })

    print("Weak-quality corruption: " + corruption_audit['mode'])
    print("Model seed: " + str(seed))
    if corruption_audit['corruption_seed'] is None:
        print("Corruption seed: none")
    else:
        print("Corruption seed: " + str(corruption_audit['corruption_seed']))
    if corruption_audit['mode'] == 'heterogeneous_gaussian':
        print("Corruption k: " + str(corruption_audit['corruption_k']))
        print("Target SNR dB: " + str(corruption_audit['target_snr_db']))
        print(
            "Total sample-view pairs: "
            + str(corruption_audit['total_sample_view_pairs'])
        )
        print(
            "Corrupted sample-view pairs: "
            + str(corruption_audit['corrupted_pair_count'])
        )
        print(
            "Per-view corruption counts: "
            + str(corruption_audit['per_view_corrupted_counts'])
        )
        print("Balanced mask: " + str(corruption_audit['balanced_mask_pass']))
        print("Mask SHA256: " + corruption_audit['mask_sha256'])
        print(
            "Global achieved SNR dB: "
            + str(corruption_audit['global_aggregate_achieved_snr_db'])
        )
        print(
            "Per-view achieved SNR dB: "
            + str(corruption_audit['per_view_aggregate_achieved_snr_db'])
        )
    if args.corruption_audit_dir is not None:
        save_corruption_audit(corruption_audit, args.corruption_audit_dir)

    print("Cluster number: " + str(n_clusters))
    view_num = len(X_list)
    print("View number: " + str(view_num))
    view_size = []
    for v in range(view_num):
        print(X_list[v].shape)
        view_size.append(X_list[v].shape[1])

    X_train = []
    for i in range(view_num):
        view_train = X_list[i]
        X_train.append(torch.from_numpy(view_train).float().to(device))

    acc_max = 0
    for ROUND in range(1):  # 10
        round_seed = seed + ROUND
        set_global_seed(round_seed)
        print("ROUND: " + str(ROUND+1))
        # Build the model
        Models = MvCAN(config, view_num, view_size, n_clusters=n_clusters, seed=round_seed, data_size=X_list[0].shape[0])
        Models.to_device(device)
        optimizers = []
        for v in range(view_num):
            optimizers.append(torch.optim.Adam(
                itertools.chain(Models.autoencoders[v].parameters()),
                lr=config['training']['lr']))
        # Print the models
        # logger.info(Models.autoencoders)
        # logger.info(optimizers)
        if Test:
            for v in range(view_num):
                checkpoint_name = config['dataset'] + str(v+1) + 'V.pth'
                checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_name))
                Models.autoencoders[v].load_state_dict(checkpoint)
            print("Loading models...")
            config['training']['init_epoch'] = 0
            config['training']['epoch'] = 0
            Models.train(config, X_train, Y_list, optimizers, device)
        else:
            # Training
            acc, nmi, ari = Models.train(config, X_train, Y_list, optimizers, device, ROUND)
            if acc > acc_max:
                acc_max = acc
                os.makedirs(args.save_dir, exist_ok=True)
                for v in range(view_num):
                    state = Models.autoencoders[v].state_dict()  # each view's model is decoupled for other views
                    checkpoint_name = config['dataset'] + str(v+1) + 'V.pth'
                    torch.save(state, os.path.join(args.save_dir, checkpoint_name))
                print('Saving...')
        if Test:
            return 0
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)

    print(accs, np.mean(accs), np.std(accs))
    print(nmis, np.mean(nmis), np.std(nmis))
    print(aris, np.mean(aris), np.std(aris))


if __name__ == '__main__':
    main()
