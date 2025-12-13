import datetime

import torch

from .data.dParser import dParser, dMixer, lMixer
from .globals import *
from .model.lDiscriminator import lDiscriminator
from .model.lGenerator import lGenerator
from .model.mGenerator import *

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = GLOBAL_LATENCY_INPUT_DIM
batch_size, epochs = 64, 5000

G = lGenerator().to(device)
D = lDiscriminator().to(device)

#dm = lMixer("data/z.csv", 0.5)
#dm.save_combination("data/cavalli_statistical_synth_latency.csv")

opt_G = torch.optim.Adam(G.parameters(), 0.0001)
opt_D = torch.optim.Adam(D.parameters(), 0.0001)
loss_fn = nn.BCELoss()

dp = dParser("data/z.csv", 0.5)
real_data = dp.all().to(device)

# Label smoothing
real_label = 0.95
fake_label = 0.05

for e in range(epochs):
    # --------------------
    # Train Discriminator
    # --------------------
    idx = torch.randint(0, len(real_data), (batch_size,))
    real = real_data[idx].to(device)

    ones = real_label * torch.ones(batch_size, 1, device=device)
    zeros = fake_label * torch.zeros(batch_size, 1, device=device)

    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)

    d_real_out = D(real)
    d_fake_out = D(fake.detach())

    d_loss = loss_fn(d_real_out, ones) + loss_fn(d_fake_out, zeros)

    opt_D.zero_grad()
    d_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
    d_grad_norm = torch.norm(
        torch.cat([p.grad.view(-1) for p in D.parameters() if p.grad is not None])
    ).item()
    opt_D.step()

    # --------------------
    # Train Generator
    # --------------------
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)
    g_out = D(fake)

    g_loss = loss_fn(g_out, ones)

    opt_G.zero_grad()
    g_loss.backward()

    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
    g_grad_norm = torch.norm(
        torch.cat([p.grad.view(-1) for p in G.parameters() if p.grad is not None])
    ).item()
    opt_G.step()

    if e % 50 == 0:
        # Compute averages
        d_real_avg = d_real_out.mean().item()
        d_fake_avg = d_fake_out.mean().item()
        d_gap = d_real_avg - d_fake_avg
        fake_std = fake.std(dim=0).mean().item()

        # Accuracy calculations
        acc_true = (d_real_out > 0.5).float().mean().item()  # fraction of real samples classified as real
        acc_false = (d_fake_out < 0.5).float().mean().item()  # fraction of fake samples classified as fake

        # Pick random examples to display (keep as tensors for computations)
        real_example = real[torch.randint(0, real.size(0), (1,))].squeeze()
        fake_example = fake[torch.randint(0, fake.size(0), (1,))].squeeze().detach()

        # Take first 3 elements only
        real_subset = real_example.cpu().numpy()
        fake_subset = fake_example.cpu().numpy()

        # Compute average absolute difference for the subset
        avg_diff = torch.abs(real_example - fake_example).mean().item()

        print(
            f"{e:4d} | "
            f"D {d_loss.item():.3f} | "
            f"G {g_loss.item():.3f} | "
            f"D_real {d_real_avg:.3f} | "
            f"D_fake {d_fake_avg:.3f} | "
            f"Gap {d_gap:.3f} | "
            f"FakeStd {fake_std:.4f} | "
            f"AccTrue {acc_true:.3f} | AccFalse {acc_false:.3f} | "
            f"Accuracy {(acc_true + acc_false) /2:.3f} |"
            f"|∇D| {d_grad_norm:.2f} | "
            f"|∇G| {g_grad_norm:.2f} | "
            f"Real3 {real_subset[:3]} | Fake3 {fake_subset[:3]} | AvgDiff {avg_diff:.4f} | "
        )

generator_class_name = G.__class__.__name__

now = datetime.datetime.now()
timestamp = now.strftime("%H%M")

save_path = f"data/models/{generator_class_name}_{timestamp}.pt"

torch.save(G.state_dict(), save_path)
print(f"Generator weights saved to: {save_path}")

full_model_path = f"{generator_class_name}_full_{timestamp}.pt"
torch.save(G, full_model_path)
print(f"Full generator model saved to: {full_model_path}")
