import torch

from globals import GLOBAL_NUM_INPUT_RANDOM_GENERATOR, GLOBAL_NUM_OUTPUT_SYNTHETIC
from model.mGenerator import *  # same class as used for training




def getSyntheticData(modelObj, modelSrc):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = GLOBAL_NUM_INPUT_RANDOM_GENERATOR
    G = modelObj.to(device)
    G.load_state_dict(torch.load(modelSrc, map_location=device))
    G.eval()
    with torch.no_grad():
        return G(torch.randn(GLOBAL_NUM_OUTPUT_SYNTHETIC, latent_dim).to(device))
