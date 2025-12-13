import datetime

import torch
import torch.nn as nn

from src.data.dParser import dParser
from src.model.mDiscriminator import *
from src.model.mGenerator import *

from globals import GLOBAL_NUM_INPUT_RANDOM_GENERATOR

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = GLOBAL_NUM_INPUT_RANDOM_GENERATOR
hidden, batch_size, epochs = 64, 64, 6000

G = mGeneratorV3_64_Residual().to(device)
D = mDiscriminatorV2().to(device)

loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), 2e-3)
opt_D = torch.optim.Adam(D.parameters(), 2e-3)

dp = dParser("data/cavalli_maha.csv", 0.01)
real_data = dp.all().to(device)

for e in range(epochs):
    idx = torch.randint(0, len(real_data), (batch_size,))
    real = real_data[idx]
    ones = torch.ones(batch_size, 1).to(device)
    zeros = torch.zeros(batch_size, 1).to(device)

    z = torch.randn(batch_size, latent_dim).to(device)
    fake = G(z)
    d_real_out = D(real)
    d_fake_out = D(fake.detach())
    d_loss = loss_fn(d_real_out, ones) + loss_fn(d_fake_out, zeros)
    opt_D.zero_grad(); d_loss.backward(); opt_D.step()

    z = torch.randn(batch_size, latent_dim).to(device)
    g_loss = loss_fn(D(G(z)), ones)
    opt_G.zero_grad(); g_loss.backward(); opt_G.step()

    if e % 400 == 0:
        d_real_avg = d_real_out.mean().item()
        d_fake_avg = d_fake_out.mean().item()
        print(f"{e:4d} | D {d_loss.item():.3f} | G {g_loss.item():.3f} "
              f"| D_real_avg {d_real_avg:.3f} | D_fake_avg {d_fake_avg:.3f}")


generator_class_name = G.__class__.__name__

now = datetime.datetime.now()
timestamp = now.strftime("%H%M")

save_path = f"data/models/{generator_class_name}_{timestamp}.pt"

torch.save(G.state_dict(), save_path)
print(f"Generator weights saved to: {save_path}")

full_model_path = f"{generator_class_name}_full_{timestamp}.pt"
torch.save(G, full_model_path)
print(f"Full generator model saved to: {full_model_path}")
