Author's code for reproducing experiments in [Behavior from the void: unsupervised active pre-training](https://arxiv.org/abs/2103.04551).
It consists of unsupervised pretraining in single Atari environment, and subsequently adapt to downstream reward function.

The code is ported out with readability in mind, it can hopeful serve as a simple starting point for future research.

## Model
![APT model diagram](./misc/APT.png)

## Usage
Run `python train.py --env_name breakout --batch_size 64 --rainbow_conv --aug --enable_cudnn --n_step 20 --start_timestep 1600 --reward_clip 1 --knn_rms --id run --proj_dim 256`

### Todo
- [ ] include hyperparameter setting as config file
- [ ] requirement file
- [ ] reward plotter
- [ ] compare against tf code
