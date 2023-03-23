import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json

import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

from clip.clip import _transform, load, tokenize
from clip.model import convert_weights, CLIP
from training.train import train, evaluate
from training.data import get_data
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import cosine_lr

# Transformer-MM-Explainability Imports
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp

def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    
    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # Do not use skip_reset unless you want to use on of the CLIP model
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(
            args.model,
            jit=False,
            is_train=True)
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIP(**model_info)
        convert_weights(model)
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)
        preprocess_test = _transform(model.visual.input_resolution, is_train=False)


    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)

    data = get_data(args, (preprocess_train, preprocess_val))

    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
        (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')
    if args.num_evals > 0:
        # Create heatmaps
        create_heatmaps_main(model, preprocess_test, args)
    elif args.train_data is None:
        evaluate(model, data, start_epoch, args, writer, 0)
        return
    elif start_epoch == 0 and args.val_data is not None:
        evaluate(model, data, 0, args, writer, 0)

    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')
        train(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        steps = data["train"].dataloader.num_batches * (epoch + 1)
        if args.val_data is not None:
            evaluate(model, data, epoch + 1, args, writer, steps)

        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (
                args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )

    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish()


# Transformer-MM-Explainability
def interpret(image, text, model, device, caption_str, output_fname, index=None):
    image_features, text_features, logit_scale = model.module(image, text)
    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    image_attn_blocks = list(dict(model.module.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    for blk in image_attn_blocks:
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = np.float32(img)
        cam = heatmap + img
        cam = cam / np.max(cam)
        return cam, img

    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis, orig_img = show_cam_on_image(image, image_relevance)
    vis, orig_img = np.uint8(255 * vis), np.uint8(255 * orig_img)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    vis_concat = np.concatenate([orig_img, vis], axis=1)
    vis_concat_captioned = add_caption_to_np_img(vis_concat, caption_str)
    plt.imsave(output_fname, vis_concat_captioned)


def add_caption_to_np_img(im_arr, caption):
    caption_img = Image.new('RGB', (im_arr.shape[1], 20), (255, 255, 255))
    # PIL.Image seems to operate on transposed axes
    d = ImageDraw.Draw(caption_img)
    d.text((5, 5), caption, fill=(0, 0, 0))
    caption_img_arr = np.uint8(np.array(caption_img))
    final_arr = np.concatenate([im_arr, caption_img_arr], axis=0)
    return final_arr


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def load_eval_ids_and_file_id_to_annotation_map(args):
    with open(args.eval_val_file, "r") as f:
        lines = f.readlines()
        val_ex = []
        for line in lines:
            val_ex.append(int(line.rstrip().split(".jpg")[0]))

    with open(args.eval_test_file, "r") as f:
        lines = f.readlines()
        test_ex = []
        for line in lines:
            test_ex.append(int(line.rstrip().split(".jpg")[0]))

    with open(args.eval_annotations_path, "r") as f:
        lines = f.readlines()
        file_id_to_annotation_map = {} # int: str
        for example in lines:
            filename, annotation = example.split("\t")
            file_id = int(filename.split(".jpg")[0]) # removes the .jpg
            if file_id in test_ex:
                file_id_to_annotation_map[file_id] = annotation.rstrip()

    np.random.seed(0)
    file_ids_to_eval = np.random.choice(list(file_id_to_annotation_map.keys()), args.num_evals)
    print("file_ids_to_eval", file_ids_to_eval)
    return file_ids_to_eval, file_id_to_annotation_map


def create_heatmaps_main(model, preprocess, args):
    device = f"cuda:{args.gpu}"
    file_ids_to_eval, file_id_to_annotation_map = load_eval_ids_and_file_id_to_annotation_map(args)

    for eval_id in tqdm(file_ids_to_eval):
        caption = file_id_to_annotation_map[eval_id]
        image_path = os.path.join(args.eval_image_dir, f"{eval_id}.jpg")
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = tokenize([caption]).to(device)
        # print(color.BOLD + color.PURPLE + color.UNDERLINE + 'text: ' + texts[0] + color.END)
        output_fname = os.path.join(args.eval_output_dir, f"{eval_id}.jpg")
        interpret(model=model, image=image, text=text, device=device, caption_str=caption, output_fname=output_fname, index=0)

    # produce local copy commands
    commands = []
    for eval_id in file_ids_to_eval:
        output_path = os.path.join(os.getcwd(), args.eval_output_dir)
        command = "scp titan1:{}/{}.jpg .".format(output_path, eval_id)
        commands.append(command)
    print("Copy commands")
    print(commands)

def main():
    args = parse_args()

    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S",
            gmtime(),
        )

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']
    #assert args.model in ['RN50', 'RN101', 'RN50x4', 'ViT-B/32'] or os.path.exists(args.model)

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()
