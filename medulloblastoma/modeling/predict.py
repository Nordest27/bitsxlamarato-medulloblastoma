import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from medulloblastoma.config import PROJ_ROOT, MODELS_DIR, PROCESSED_DATA_DIR
import os
import numpy as np
import pandas as pd
import torch
from medulloblastoma.modeling.train_model import *
from medulloblastoma.modeling.my_model import *
from joblib import load

os.chdir(PROJ_ROOT)

def check_data(data):
    # check if data is DataFrame or numpy array
    if isinstance(data, pd.DataFrame):
        data = torch.tensor(data.values).float()
    elif isinstance(data, np.ndarray):
        data = torch.tensor(data).float()
    elif isinstance(data, torch.Tensor):
        data = data.float()
    else:
        raise ValueError("data is neither a pandas DataFrame, a numpy array, nor a torch tensor")
    return data

def apply_VAE(data,model_here,y=None):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data2 = scaler.transform(data)
    
    with torch.no_grad():
        if y is None:
            data_latent, mu, logvar, z = model_here(torch.tensor(data2).float())
            data_vae = scaler.inverse_transform(data_latent)
        else:
            data_latent, mu, logvar, z = model_here(torch.tensor(data2).float(),torch.tensor(y).float())
            data_vae = scaler.inverse_transform(data_latent)

    return data_vae, mu, logvar, z, scaler

def load_model(model_path,model,hyperparams,seed=2023):
    # Importing the model:
    set_seed(seed)
    if model.__name__ == 'VAE':
        idim, md, feat = hyperparams
        model_vae = model(input_dim=idim, mid_dim=md, features=feat)  # Initialize the model
    elif model.__name__ == 'NetworkReconstruction':
        model_vae = model(hyperparams)
    else:
        raise ValueError('Model not recognized. Must be VAE or NetworkReconstruction')
    model_vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
    model_vae.eval()  # Set the model to evaluation mode
    return model_vae


def predict(data, model_path, mid_dim, features, logreg_path):
    if 'statistical' in model_path:
        input_dim = 14403
    else:
        input_dim = 2886

    model_vae = load_model(
        # MAHA
        model_path=model_path,
        hyperparams=(input_dim, mid_dim, features),
        # STATISTICAL
        model=VAE,
    )
    data_for_model = check_data(data=data)

    decoded, mu, logvar, z, scaler = apply_VAE(data=data_for_model, model_here=model_vae, y=None)
    df_z = pd.DataFrame(z, index=data.index)

    mu = mu.cpu().numpy()
    logvar = logvar.cpu().numpy()

    X_mu = mu[df_z.index.get_indexer(data.index)]

    logreg = load(logreg_path)

    p = logreg.predict_proba(X_mu)[:, 1]

    # Return score
    return pd.Series(index=data.index, data=p)

# Example of use
# data=pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'g3g4_statistical.csv'), index_col=0)
# model_path = os.path.join(MODELS_DIR, 'statistical', 'best_model.pth')
# logreg_path = os.path.join(MODELS_DIR, 'statistical', 'logreg_save.joblib')
# predict(data=data, model_path=model_path, mid_dim=1024, features=32, logreg_path=logreg_path)