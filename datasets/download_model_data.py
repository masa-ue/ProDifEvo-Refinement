import wandb
import os
run = wandb.init()


artifact = run.use_artifact("fderc_diffusion/Inverse_PF/AlphaFoldPDB:v0")
dir = artifact.download()
os.system('tar -xvzf artifacts/AlphaFoldPDB:v0/AlphaFoldPDB.tar.gz')

wandb.finish()
