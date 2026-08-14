import os
import argparse
import itertools
import json
import torch
import random
import numpy as np
from model import MvCAN
from util import get_logger
from datasets import *
from configure import get_default_config
from weak_quality import apply_weak_quality_protocol, save_corruption_audit
from irv.b3_audit import hash_backbone, hash_semantic_heads
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
parser.add_argument(
    '--semantic_mode',
    choices=('off', 'detached', 'uniform'),
    default='off',
)
parser.add_argument('--semantic_dim', type=int, default=None)
parser.add_argument('--semantic_seed', type=int, default=None)
parser.add_argument('--semantic_lr', type=float, default=1e-4)
parser.add_argument('--semantic_temperature', type=float, default=0.2)
parser.add_argument('--semantic_audit_dir', type=str, default=None)
parser.add_argument('--semantic_save_dir', type=str, default=None)
args = parser.parse_args()
dataset = dataset[args.dataset]


def save_b3_a0_audit(
    audit_dir,
    config,
    model_seed,
    corruption_audit,
    models,
    acc,
    nmi,
    ari,
):
    """Save the detached-semantic protection audit without serializing tensors."""
    noisy = corruption_audit['mode'] != 'none'
    semantic_enabled = models.semantic_heads is not None
    audit = {
        'stage': 'B3-A0',
        'dataset': config['dataset'],
        'model_seed': int(model_seed),
        'corruption_protocol_version': corruption_audit.get('protocol_version'),
        'corruption_mode': corruption_audit['mode'],
        'corruption_seed': corruption_audit.get('corruption_seed'),
        'corruption_k': corruption_audit.get('corruption_k') if noisy else None,
        'target_snr_db': corruption_audit.get('target_snr_db') if noisy else None,
        'corruption_mask_sha256': (
            corruption_audit.get('mask_sha256') if noisy else None
        ),
        'semantic_mode': models.semantic_mode,
        'semantic_enabled': bool(semantic_enabled),
        'latent_dim': int(models._latent_dim),
        'semantic_dim': models.semantic_dim,
        'semantic_seed': models.semantic_seed,
        'view_num': int(models.view_num),
        'native_metrics': {
            'acc': float(acc),
            'nmi': float(nmi),
            'ari': float(ari),
        },
        'backbone_hash': hash_backbone(models.autoencoders),
        'semantic_hash': hash_semantic_heads(models.semantic_heads),
        'runtime': dict(models.semantic_runtime_audit),
    }
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = os.path.join(audit_dir, 'b3_a0_audit.json')
    with open(audit_path, 'w') as audit_file:
        json.dump(audit, audit_file, indent=2, sort_keys=True)
        audit_file.write('\n')
    print('B3 audit: ' + audit_path)


def save_b3_a1_audit(
    audit_dir,
    config,
    model_seed,
    corruption_audit,
    models,
    acc,
    nmi,
    ari,
):
    """Save B3-A1 backbone protection and semantic update diagnostics."""
    noisy = corruption_audit['mode'] != 'none'
    semantic_enabled = models.semantic_heads is not None
    audit = {
        'stage': 'B3-A1',
        'dataset': config['dataset'],
        'model_seed': int(model_seed),
        'corruption_protocol_version': corruption_audit.get('protocol_version'),
        'corruption_mode': corruption_audit['mode'],
        'corruption_seed': corruption_audit.get('corruption_seed'),
        'corruption_k': corruption_audit.get('corruption_k') if noisy else None,
        'target_snr_db': corruption_audit.get('target_snr_db') if noisy else None,
        'corruption_mask_sha256': (
            corruption_audit.get('mask_sha256') if noisy else None
        ),
        'semantic_mode': models.semantic_mode,
        'semantic_enabled': bool(semantic_enabled),
        'latent_dim': int(models._latent_dim),
        'semantic_dim': models.semantic_dim,
        'semantic_seed': models.semantic_seed,
        'semantic_lr': models.semantic_lr,
        'semantic_temperature': models.semantic_temperature,
        'view_num': int(models.view_num),
        'native_metrics': {
            'acc': float(acc),
            'nmi': float(nmi),
            'ari': float(ari),
        },
        'backbone_hash': hash_backbone(models.autoencoders),
        'semantic_hash_initial': models.semantic_hash_initial,
        'semantic_hash_final': models.semantic_hash_final,
        'runtime': dict(models.semantic_runtime_audit),
    }
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = os.path.join(audit_dir, 'b3_a1_audit.json')
    with open(audit_path, 'w') as audit_file:
        json.dump(audit, audit_file, indent=2, sort_keys=True)
        audit_file.write('\n')
    print('B3-A1 audit: ' + audit_path)


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

    semantic_config = None
    semantic_dim = None
    semantic_seed = None
    semantic_lr = None
    semantic_temperature = None
    if args.semantic_mode in ('detached', 'uniform'):
        semantic_dim = (
            config['Autoencoder']['arch'][-1]
            if args.semantic_dim is None
            else int(args.semantic_dim)
        )
        semantic_seed = (
            seed + 1000
            if args.semantic_seed is None
            else int(args.semantic_seed)
        )
        semantic_config = {
            'mode': args.semantic_mode,
            'semantic_dim': semantic_dim,
            'semantic_seed': semantic_seed,
        }
        if args.semantic_mode == 'uniform':
            semantic_lr = float(args.semantic_lr)
            semantic_temperature = float(args.semantic_temperature)
            semantic_config.update({
                'semantic_lr': semantic_lr,
                'semantic_temperature': semantic_temperature,
            })

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
    print("B3 semantic mode: " + args.semantic_mode)
    print("B3 semantic dim: " + (str(semantic_dim) if semantic_dim is not None else "none"))
    print("B3 semantic seed: " + (str(semantic_seed) if semantic_seed is not None else "none"))
    print(
        "B3 semantic learning rate: "
        + (str(semantic_lr) if semantic_lr is not None else "none")
    )
    print(
        "B3 semantic temperature: "
        + (
            str(semantic_temperature)
            if semantic_temperature is not None
            else "none"
        )
    )
    print(
        "B3 semantic backbone protection: "
        + (
            "detached"
            if args.semantic_mode in ('detached', 'uniform')
            else "semantic off"
        )
    )
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
        Models = MvCAN(
            config,
            view_num,
            view_size,
            n_clusters=n_clusters,
            seed=round_seed,
            data_size=X_list[0].shape[0],
            semantic_config=semantic_config,
        )
        Models.to_device(device)
        optimizers = []
        for v in range(view_num):
            optimizers.append(torch.optim.Adam(
                itertools.chain(Models.autoencoders[v].parameters()),
                lr=config['training']['lr']))
        semantic_optimizer = None
        if args.semantic_mode == 'uniform':
            semantic_optimizer = torch.optim.Adam(
                Models.semantic_heads.parameters(),
                lr=semantic_lr,
            )
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
            Models.train(
                config,
                X_train,
                Y_list,
                optimizers,
                device,
                semantic_optimizer=semantic_optimizer,
            )
        else:
            # Training
            acc, nmi, ari = Models.train(
                config,
                X_train,
                Y_list,
                optimizers,
                device,
                ROUND,
                semantic_optimizer=semantic_optimizer,
            )
            if acc > acc_max:
                acc_max = acc
                os.makedirs(args.save_dir, exist_ok=True)
                for v in range(view_num):
                    state = Models.autoencoders[v].state_dict()  # each view's model is decoupled for other views
                    checkpoint_name = config['dataset'] + str(v+1) + 'V.pth'
                    torch.save(state, os.path.join(args.save_dir, checkpoint_name))
                print('Saving...')
            if (
                args.semantic_mode == 'uniform'
                and args.semantic_save_dir is not None
            ):
                os.makedirs(args.semantic_save_dir, exist_ok=True)
                semantic_checkpoint_path = os.path.join(
                    args.semantic_save_dir,
                    'semantic_heads.pth',
                )
                torch.save(
                    Models.semantic_heads.state_dict(),
                    semantic_checkpoint_path,
                )
                print('Saving semantic heads: ' + semantic_checkpoint_path)
        if Test:
            return 0
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)

    if args.semantic_audit_dir is not None:
        save_b3_a1_audit(
            args.semantic_audit_dir,
            config,
            seed,
            corruption_audit,
            Models,
            acc,
            nmi,
            ari,
        )

    print(accs, np.mean(accs), np.std(accs))
    print(nmis, np.mean(nmis), np.std(nmis))
    print(aris, np.mean(aris), np.std(aris))


if __name__ == '__main__':
    main()
