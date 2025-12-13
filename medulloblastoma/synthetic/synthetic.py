from .model.mGenerator import *


def getSyntheticData(modelObj, modelSrc, num):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = modelObj.to(device)
    G.load_state_dict(torch.load(modelSrc, map_location=device))
    G.eval()
    with torch.no_grad():
        return G(torch.randn(num, GLOBAL_NUM_INPUT_RANDOM_GENERATOR).to(device)).to("cpu")
