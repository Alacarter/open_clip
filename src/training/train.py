import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

from .zero_shot import zero_shot_eval

import sys
import pdb
# import wandb

import logging

def is_master(args):
    return (not args.distributed) or args.gpu == 0


def get_logits(model, images, texts, args):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    return logits_per_image, logits_per_text, ground_truth


def get_loss(model, images, texts, loss_img, loss_txt, args):
    logits_per_image, logits_per_text, ground_truth = get_logits(
        model, images, texts, args)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2
    return total_loss


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def get_log_probs(model, images, texts, args):
    logits_per_image, logits_per_text, ground_truth = get_logits(
        model, images, texts, args)
    # logits are shape (bsz, bsz). logits_per_text is logits_per_image.T.
    log_probs_per_image = nn.functional.log_softmax(logits_per_image, dim=1)
    log_probs_per_text = nn.functional.log_softmax(logits_per_text, dim=1)
    return log_probs_per_image, log_probs_per_text


def get_distillation_loss(
        teacher_model, student_model, teacher_images, teacher_texts,
        student_images, student_texts, loss_img, loss_txt, args):
    teacher_log_probs_per_image, teacher_log_probs_per_text = get_log_probs(teacher_model, teacher_images, teacher_texts, args)
    student_log_probs_per_image, student_log_probs_per_text = get_log_probs(student_model, student_images, student_texts, args)

    image_KL = loss_img(student_log_probs_per_image, teacher_log_probs_per_image)
    text_KL = loss_txt(student_log_probs_per_text, teacher_log_probs_per_text)

    distillation_loss = 0.5 * (image_KL + text_KL)
    return distillation_loss, image_KL, text_KL


def save_metrics(
        all_image_features, all_text_features,
        cumulative_loss, num_elements, epoch, zero_shot_metrics,
        tb_writer, args, results_fname="results.jsonl", prefix="",
        cum_image_loss=None, cum_text_loss=None):
    with torch.no_grad():
        metrics = get_metrics(
                torch.cat(all_image_features), torch.cat(all_text_features)
            )
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )

        if cum_image_loss is not None:
            image_loss = cum_image_loss / num_elements
            metrics.update({"val_image_loss": image_loss.item()})
        if cum_text_loss is not None:
            text_loss = cum_text_loss / num_elements
            metrics.update({"val_text_loss": text_loss.item()})

        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{prefix}_{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, results_fname), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def train_distillation(teacher_model, student_model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)

    teacher_model.eval()
    student_model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.KLDivLoss(log_target=True, reduction='batchmean')
    loss_txt = nn.KLDivLoss(log_target=True, reduction='batchmean')
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        teacher_images, teacher_texts, student_images, student_texts = batch
        if args.gpu is not None:
            teacher_images = teacher_images.cuda(args.gpu, non_blocking=True)
            teacher_texts = teacher_texts.cuda(args.gpu, non_blocking=True)
            student_images = student_images.cuda(args.gpu, non_blocking=True)
            student_texts = student_texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m_teacher = teacher_model.module if args.distributed or args.dp else teacher_model
        m_student = student_model.module if args.distributed or args.dp else student_model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss, total_image_loss, total_text_loss = get_distillation_loss(
                    teacher_model, student_model, teacher_images, teacher_texts,
                    student_images, student_texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss, total_image_loss, total_text_loss = get_distillation_loss(
                teacher_model, student_model, teacher_images, teacher_texts,
                student_images, student_texts, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m_teacher.logit_scale.data = torch.clamp(m_teacher.logit_scale.data, 0, 4.6052)
        m_student.logit_scale.data = torch.clamp(m_student.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(student_images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m_student.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m_student.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None, results_fname="results.jsonl", distill_model_type=""):
    if not is_master(args):
        return
    
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in dataloader:
            if args.distillation:
                if distill_model_type == "teacher":
                    images, texts, _, _ = batch
                elif distill_model_type == "student":
                    _, _, images, texts = batch
            else:
                images, texts = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

    metrics = save_metrics(
        all_image_features, all_text_features,
        cumulative_loss, num_elements, epoch, zero_shot_metrics,
        tb_writer, args, results_fname=results_fname, prefix=distill_model_type)

    return metrics


def evaluate_distillation(teacher_model, student_model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return

    teacher_model.eval()
    student_model.eval()

    zero_shot_metrics = zero_shot_eval(student_model, data, epoch, args)

    dataloader = data['val'].dataloader

    loss_img = nn.KLDivLoss(log_target=True, reduction='batchmean')
    loss_txt = nn.KLDivLoss(log_target=True, reduction='batchmean')
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    cum_image_loss = 0.0
    cum_text_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in dataloader:
            teacher_images, teacher_texts, student_images, student_texts = batch
            if args.gpu is not None:
                teacher_images = teacher_images.cuda(args.gpu, non_blocking=True)
                teacher_texts = teacher_texts.cuda(args.gpu, non_blocking=True)
                student_images = student_images.cuda(args.gpu, non_blocking=True)
                student_texts = student_texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = student_model(student_images, student_texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(student_images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss, total_image_loss, total_text_loss = get_distillation_loss(
                teacher_model, student_model, teacher_images, teacher_texts,
                student_images, student_texts, loss_img, loss_txt, args)

            batch_size = len(student_images)
            cumulative_loss += total_loss * batch_size
            cum_image_loss += total_image_loss * batch_size
            cum_text_loss += total_text_loss * batch_size
            num_elements += batch_size

    metrics = save_metrics(
        all_image_features, all_text_features,
        cumulative_loss, num_elements, epoch, zero_shot_metrics,
        tb_writer, args, cum_image_loss=cum_image_loss, cum_text_loss=cum_text_loss)

    return metrics


def get_metrics(image_features, text_features):
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = (
        torch.arange(len(text_features)).view(-1, 1).to(logits_per_image.device)
    )

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
