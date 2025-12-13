import torch

from src.globals import GLOBAL_NUM_INPUT_RANDOM_GENERATOR, GLOBAL_NUM_OUTPUT_SYNTHETIC
from src.model.mGenerator import mGeneratorV1_2886  # same class as used for training


device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = GLOBAL_NUM_INPUT_RANDOM_GENERATOR
G = mGeneratorV1_2886().to(device)
G.load_state_dict(torch.load("mGeneratorV1_2886_1430.pt", map_location=device))
G.eval()
with torch.no_grad():
    synthetic_data = G(torch.randn(GLOBAL_NUM_OUTPUT_SYNTHETIC, latent_dim).to(device))