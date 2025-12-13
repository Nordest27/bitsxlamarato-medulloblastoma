import datetime

import torch
import torch.nn as nn

from src.data.dParser import dParser, dMixer
from src.model.mDiscriminator import *
from src.model.mGenerator import *

from globals import GLOBAL_NUM_INPUT_RANDOM_GENERATOR

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = GLOBAL_NUM_INPUT_RANDOM_GENERATOR
hidden, batch_size, epochs = 64, 64, 1000

G = mGeneratorV4_Residual().to(device)
D = mDiscriminatorV3().to(device)


dm = dMixer("data/cavalli_statistical.csv", 0.5)
dm.combination()
dm.save_combination("data/cavalli_statistical_synth.csv")



loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), 2e-3)
opt_D = torch.optim.Adam(D.parameters(), 2e-3)

dp = dParser("data/cavalli_statistical.csv", 0.01)
real_data = dp.all().to(device)

for e in range(epochs):
    idx = torch.randint(0, len(real_data), (batch_size,))
    real = real_data[idx].to(device)

    ones = torch.ones(batch_size, 1, device=device)
    zeros = torch.zeros(batch_size, 1, device=device)

    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)

    d_real_out = D(real)
    d_fake_out = D(fake.detach())

    d_loss = loss_fn(d_real_out, ones) + loss_fn(d_fake_out, zeros)

    opt_D.zero_grad()
    d_loss.backward()
    opt_D.step()

    z = torch.randn(batch_size, latent_dim, device=device)
    g_fake = G(z)
    g_out = D(g_fake)

    g_loss = loss_fn(g_out, ones)

    opt_G.zero_grad()
    g_loss.backward()
    opt_G.step()

    for e in range(epochs):
        idx = torch.randint(0, len(real_data), (batch_size,))
        real = real_data[idx].to(device)

        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)

        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)

        d_real_out = D(real)
        d_fake_out = D(fake.detach())

        d_loss = loss_fn(d_real_out, ones) + loss_fn(d_fake_out, zeros)

        opt_D.zero_grad()
        d_loss.backward()
        d_grad_norm = torch.norm(
            torch.cat([p.grad.view(-1) for p in D.parameters() if p.grad is not None])
        ).item()
        opt_D.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)
        g_out = D(fake)

        g_loss = loss_fn(g_out, ones)

        opt_G.zero_grad()
        g_loss.backward()
        g_grad_norm = torch.norm(
            torch.cat([p.grad.view(-1) for p in G.parameters() if p.grad is not None])
        ).item()
        opt_G.step()

        d_real_avg = d_real_out.mean().item()
        d_fake_avg = d_fake_out.mean().item()
        d_gap = d_real_avg - d_fake_avg
        fake_std = fake.std(dim=0).mean().item()

        if e % 10 == 0:
            print(
                f"{e:4d} | "
                f"D {d_loss.item():.3f} | "
                f"G {g_loss.item():.3f} | "
                f"D_real {d_real_avg:.3f} | "
                f"D_fake {d_fake_avg:.3f} | "
                f"Gap {d_gap:.3f} | "
                f"FakeStd {fake_std:.4f} | "
                f"|∇D| {d_grad_norm:.2f} | "
                f"|∇G| {g_grad_norm:.2f}"
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
