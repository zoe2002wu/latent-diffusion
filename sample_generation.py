#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model


def main(args):
    model = get_model()
    sampler = DDIMSampler(model)

    batch_size = args.batch_size
    num_samples = args.num_samples
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    penalty_param = args.penalty_param
    if args.continue_path:
        continue_path = args.continue_path
        data = np.load(continue_path)
        all_images = data['arr_0'].tolist()      # images
        all_labels = data['arr_1'].tolist()  # labels
        print(f"Resuming from {len(all_images)} samples in {continue_path}")
    else:
        all_images = []
        all_labels = []

    step = 0
    with torch.no_grad():
        with model.ema_scope():
            # unconditional conditioning
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(batch_size * [1000]).to(model.device)}
            )

            classes = torch.randint(0, 1000, (batch_size,)).to(model.device)
            
            while len(all_images) < num_samples:
                c = model.get_learned_conditioning(
                    {model.cond_stage_key: classes.to(model.device)}
                )

                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=[3, 64, 64],
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    riemann=1,
                    penalty_param=penalty_param,
                    eta=ddim_eta,
                )

                # decode
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)

                # format for saving
                sample = ((x_samples_ddim + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                sample = sample.permute(0, 2, 3, 1).contiguous()

                all_images.extend([s.cpu().numpy() for s in sample])
                all_labels.extend(classes.cpu().numpy().tolist())
                print(f"Created {len(all_images)} samples so far...")

                classes = torch.randint(0, 1000, (batch_size,)).to(model.device)
            
                if step % (num_samples//(4*batch_size)) == 0 or len(all_images)>=num_samples:
                    arr = np.stack(all_images, axis=0)
                    arr = arr[:num_samples]
                    label_arr = all_labels[:num_samples]

                    # shape_str = "x".join([str(x) for x in arr.shape])
                    out_path = f"samples_{num_samples}_{penalty_param}_{scale}_{ddim_steps}_{batch_size}.npz"
                    print(f"Saving to {out_path}")
                    np.savez(out_path, arr, label_arr)

                    print("Sampling complete!")

                step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--ddim_eta", type=float)
    parser.add_argument("--penalty_param", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--continue_path", type=str)
    args = parser.parse_args()

    main(args)







