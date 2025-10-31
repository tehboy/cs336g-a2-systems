import logging
import numpy as np
import pickle
import random
import sys
import time
import torch
import wandb

from cs336_basics.basics import TransformerLanguageModel
from cs336_basics.training import AdamW, get_lr_cosine_schedule, cross_entropy, gradient_clipping
from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.utils import ModelArgs, get_device
import os


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )


def main():
    start_time = time.time()
    args = ModelArgs()
    setup_logging()
    logging.info(f"Training file path: {args.bpe_path}")

    # Parse device and dtype
    device = get_device()
    dtype = getattr(torch, str(args.dtype))

    # Torch
    torch.manual_seed(int(args.seed))
    # NumPy
    np.random.seed(int(args.seed))
    # Python
    random.seed(int(args.seed))

    # Initialize wandb
    run = wandb.init(
        entity="natjambo",
        project="cs336",
        config={
            "train_file": args.bpe_path,
            "valid_file": args.valid_bpe_path,
            "vocab_size": args.vocab_size,
            "training_steps": args.training_steps,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "dtype": args.dtype,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "eps": args.eps,
            "seed": args.seed,
        },
    )
    # Initialize TransformerLanguageModel
    model = TransformerLanguageModel(
        vocab_size=int(args.vocab_size),
        context_length=int(args.context_length),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        d_ff=int(args.d_ff),
        rope_theta=float(args.rope_theta),
        device=device,
        dtype=dtype,
        use_norms=(not args.ablate_norms),
        use_rope=(not args.ablate_position_embeddings),
        use_silu=args.use_silu,
    )
    model.train()
    model.to(device)
    logging.info(
        f"Initialized TransformerLanguageModel with vocab_size={args.vocab_size}, context_length={args.context_length}, d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}, rope_theta={args.rope_theta}, device={device}, dtype={dtype}"
    )

    # Initialize AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.max_lr),
        weight_decay=float(args.weight_decay),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.eps),
    )
    logging.info(
        f"Initialized AdamW optimizer with weight_decay={args.weight_decay}, betas=({args.beta1}, {args.beta2}), eps={args.eps}"
    )

    if args.load_checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint_file}")
        t = load_checkpoint(str(args.checkpoint_file), model, optimizer)
    else:
        t = 1

    best_validation_loss = float("inf")
    best_validation_t = None
    if os.path.exists(str(args.validation_checkpoint_file)):
        logging.info(
            f"Validation checkpoint file {args.validation_checkpoint_file} exists. Loading best validation loss."
        )
        validation_checkpoint = torch.load(args.validation_checkpoint_file)
        best_validation_loss = validation_checkpoint["loss"]
        best_validation_t = validation_checkpoint["iteration"]
        logging.info(
            f"Loaded best validation loss: {best_validation_loss} at t: {best_validation_t}"
        )

    with open(str(args.bpe_shape_path), "rb") as f:
        train_file_shape = pickle.load(f)
    train_file = np.memmap(args.bpe_path, dtype=np.uint16, mode="r", shape=train_file_shape)
    logging.info(
        f"Memmapped training file: {args.bpe_path} with shape {train_file.shape} and dtype {train_file.dtype}"
    )

    with open(str(args.valid_bpe_shape_path), "rb") as f:
        valid_file_shape = pickle.load(f)
    valid_file = np.memmap(args.valid_bpe_path, dtype=np.uint16, mode="r", shape=valid_file_shape)
    logging.info(
        f"Memmapped validation file: {args.valid_bpe_path} with shape {valid_file.shape} and dtype {valid_file.dtype}"
    )

    logging.info("Beginning training run.")
    run.watch(model, log="all", log_freq=100)
    while t <= int(args.training_steps):
        iter_start_time = time.time()

        # Update learning rate
        lr = get_lr_cosine_schedule(
            t,
            float(args.max_lr),
            float(args.min_lr),
            int(args.warmup_iters),
            int(args.cosine_cycle_iters),
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        batch_losses = []
        batch_times = []
        batch_grads = []
        for batch_n in range(int(args.num_batches)):
            batch_start_time = time.time()
            input, target = get_batch(
                train_file, int(args.batch_size), int(args.context_length), device
            )
            batch_end_time = time.time()
            optimizer.zero_grad()

            # Forward pass
            forward_start_time = time.time()
            output = model(input)
            forward_end_time = time.time()
            # Compute loss
            loss_start_time = time.time()
            loss = cross_entropy(output, target)
            loss_end_time = time.time()
            # Backward pass
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            # Gradient clipping
            gradient_clip_start_time = time.time()
            batch_grads.append(gradient_clipping(model.parameters(), 1.0))
            gradient_clip_end_time = time.time()
            # Update weights
            optimizer_step_start_time = time.time()
            optimizer.step()
            optimizer_step_end_time = time.time()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            batch_times.append(time.time() - batch_start_time)
            if t % 10 == 0 or t == 1:
                logging.info(f"Step {t} Batch {batch_n}: loss={loss.item():.4f}, lr={lr:.6f}")
                logging.info(f"  get_batch time: {batch_end_time - batch_start_time:.4f}s")
                logging.info(f"  Forward pass time: {forward_end_time - forward_start_time:.4f}s")
                logging.info(f"  Loss calculation time: {loss_end_time - loss_start_time:.4f}s")
                logging.info(
                    f"  Backward pass time: {backward_end_time - backward_start_time:.4f}s"
                )
                logging.info(
                    f"  Gradient clip time: {gradient_clip_end_time - gradient_clip_start_time:.4f}s"
                )
                logging.info(
                    f"  Optimizer step time: {optimizer_step_end_time - optimizer_step_start_time:.4f}s"
                )

        # Use average loss and time for logging
        running_loss = torch.tensor(batch_losses).mean()
        g = torch.tensor(batch_grads).mean()
        batch_time = sum(batch_times) / len(batch_times)
        to_log = {
            "epoch": t,
            "lr": lr,
            "avg_grad_norm": g,
            "avg_epoch_loss": running_loss.item(),
            "avg_batch_time": batch_time,
        }
        if t % args.validation_interval == 0:
            validation_start_time = time.time()
            model.eval()
            with torch.no_grad():
                validation_loss = 0.0
                for _ in range(args.num_validation_batches):
                    validation_input, validation_target = get_batch(
                        valid_file, int(args.batch_size), int(args.context_length), device
                    )

                    validation_output = model(validation_input)
                    validation_loss += cross_entropy(validation_output, validation_target).item()
                avg_validation_loss = validation_loss / args.num_validation_batches
                if avg_validation_loss < best_validation_loss:
                    best_validation_loss = avg_validation_loss
                    best_validation_t = t
                    save_checkpoint(
                        model, optimizer, t, args.validation_checkpoint_file, best_validation_loss
                    )
                logging.info(f"Step {t} Validation : loss={avg_validation_loss:.4f}")
                to_log["validation_loss"] = avg_validation_loss
                to_log["best_validation_loss"] = best_validation_loss
                to_log["best_validation_t"] = best_validation_t
            model.train()
            validation_end_time = time.time()
            logging.info(
                f"  Validation step time: {validation_end_time - validation_start_time:.4f}s"
            )

        run.log(to_log, step=t)
        iter_end_time = time.time()

        # Logging
        if t % args.checkpoint_interval == 1:
            checkpoint_start = time.time()
            save_checkpoint(model, optimizer, t, args.checkpoint_file)
            checkpoint_elapsed = time.time() - checkpoint_start
            logging.info(f"Checkpointing at step {t} (took {checkpoint_elapsed:.2f} seconds)")
        if t % 10 == 0 or t == 1:
            logging.info(f"  Total iteration time: {iter_end_time - iter_start_time:.4f}s")
            logging.info(f"  Running loss: {running_loss:.4f}")
            logging.info(f"  Average batch time: {batch_time:.4f}s")
        t += 1
    # Final save
    save_checkpoint(model, optimizer, t, args.checkpoint_file)
    run.finish()
    elapsed = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")