import argparse
import os


def train_args(string=None):
    parser = argparse.ArgumentParser(description="Video Walk Training")

    parser.add_argument("--exp_name", default="", type=str, help="Experiment name")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank; for cpu only, use -1",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for initialization"
    )

    ## Data paths args
    parser.add_argument(
        "--data_path",
        default="datasets/movi/",
        help="Dataset path",
    )

    ## Data args
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--data_aug_setting",
        default="setting2",
        help="Select between data augmentation setting setting1, setting2, setting3",
    )

    ## Training args
    parser.add_argument("-b", "--per_gpu_train_batch_size", default=16, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)

    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    ## Model args

    parser.add_argument(
        "--pretrained_gmflow",
        default=False,
        action="store_true",
        help="Use pretrained GMFlow weights",
    )
    parser.add_argument(
        "--pretrained_gmflow_path",
        default="",
        type=str,
        help="Path to pretrained GMFlow model gmflow_things.pth",
    )

    parser.add_argument(
        "--temp",
        default=0,
        type=float,
        help="softmax temperature when computing affinity",
    )

    ### GMFlow args ###

    ###################


    parser.add_argument(
        "--use_amp",
        default=False,
        action="store_true",
        help="Use Auto Mixed Precision for training",
    )

    parser.add_argument(
        "--use_flash_attention",
        default=False,
        action="store_true",
        help="Use Flash Attention",
    )

    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Use Gradient Checkpointing",
    )

    
    parser.add_argument(
        "--smoothness_edge_constant",
        default=150.0,
        type=float,
        help="Edge constant in Smoothness loss",
    )

    parser.add_argument(
        "--smoothness_loss_weight",
        default=1,
        type=float,
        help="Weight for smoothness loss",
    )


    parser.add_argument(
        "--smoothness_curriculum",
        default=False,
        action="store_true",
        help="Use curriculum for Smoothness Loss Lambda",
    )

  
    parser.add_argument(
        "--use_data_aug_curriculum",
        default=False,
        action="store_true",
        help="Use curriculum for Random Resize Data Augmentation",
    )

    parser.add_argument(
        "--smoothness_loss",
        default=False,
        action="store_true",
        help="Enable Smoothness Loss",
    )

    parser.add_argument(
        "--no_of_frames",
        default=2,
        type=int,
        help="No. of frames to use in a cycle",
    )

    # Optim args

    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")

    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    ## Logging args
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--fast_test", default=False, action="store_true", help="Debug"
    )

    ## Viz args
    parser.add_argument(
        "--visualize", default=False, action="store_true", help="visualize with wandb"
    )

    parser.add_argument(
        "--max_viz_per_batch",
        default=20,
        type=int,
        help="Max instances to visualize per batch",
    )


    parser.add_argument(
        "--train_viz_log_freq",
        default=50,
        type=int,
        help="Viz log frequency during training",
    )

    parser.add_argument(
        "--model_save_freq",
        default=1000,
        type=int,
        help="Viz log frequency during training",
    )

    parser.add_argument(
        "--compute_flow_method",
        default="weighted_sum",
        help='Affinity matrix to Flow computation method. Choose from ["argmax", "weighted_sum"]',
    )
    parser.add_argument(
        "--window_threshold",
        default=5,
        type=int,
        help="Threshold for weighted argmax",
    )
    
    ## Load from checkpoint args
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    ## Debug args
    parser.add_argument(
        "--disable_transforms",
        dest="disable_transforms",
        help="Disable dataset transforms for testing",
        action="store_true",
    )

    # Augmentation args

    parser.add_argument(
        "--random_frame_skip",
        default=False,
        help="Random frame skip between 1 and frame_skip",
        action="store_true",
    )
    parser.add_argument("--img_size1_h", default=256, type=int)
    parser.add_argument("--img_size1_w", default=256, type=int)
    parser.add_argument("--img_size2_h", default=256, type=int)
    parser.add_argument("--img_size2_w", default=256, type=int)


    parser.add_argument(
        "--eval_only",
        dest="eval_only",
        help="",
        action="store_true",
    )

    if string is not None:
        args = parser.parse_args(string.split())
    else:
        args = parser.parse_args()

    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    except:
        args.local_rank = -1

    if args.fast_test:
        args.batch_size = 1
        args.workers = 0
        args.local_rank = -1

    if args.exp_name == "":
        import datetime

        dt = datetime.datetime.today()
        args.exp_name = f"{str(dt.year)}-{str(dt.month)}-{str(dt.day)}-{str(dt.hour)}-{str(dt.minute)}-{str(dt.second)}"

    args.output_dir = f"results/{args.exp_name}"
    
    if "kinetics" in args.data_path.lower():
        args.dataset_type = "kinetics"
    elif "movi" in args.data_path.lower():
        args.dataset_type = "kubric"
    else:
        raise NotImplementedError("Dataset not implemented")

    if args.pretrained_gmflow_path == "":
        args.pretrained_gmflow_path = (
            "pretrained_models/gmflow_with_refine_things-36579974.pth"
        )

    html_filename = "transform_label"

    assert args.data_aug_setting in ["setting0", "setting1", "setting2", "setting3", "setting4"]

    args.html_filename = html_filename

    args.img2feature_downsample = 4

    return args
