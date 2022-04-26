# Gromov-Wasserstein Cross Domain Imitation Learning

This is the official PyTorch implementation of the ICLR 2022 paper [Cross-Domain Imitation Learning via Optimal Transport](https://arxiv.org/abs/2110.03684).

If you use this code in your research project please cite us as:
```
@inproceedings{fickinger2022gromov,
  title={Cross-Domain Imitation Learning via Optimal Transport},
  author={Fickinger, Arnaud and Cohen, Samuel and Russell, Stuart and Amos, Brandon},
  booktitle={10th International Conference on Learning Representations, ICLR},
  year={2022}
}
```

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment and activate it:
```
conda env create -f conda_env.yml
source activate gwil
```

## Instructions

Expert demonstrations are available [here](https://drive.google.com/file/d/1xE882IuQkXUuaeXHInYaP9eqvhQm48Et/view?usp=sharing). Copy the directory exp at the root of this repo.

### Training the expert policies

Only needed if new expert demonstrations are needed. The parameter num_train_steps is set such that the policy obtained is approximately optimal in the environment as observed in [this repo's result plots](https://github.com/denisyarats/pytorch_sac).

```
python train.py env=pendulum_swingup num_train_steps=1e6 experiment=expert
python train.py env=cartpole_swingup num_train_steps=5e5 experiment=expert
python train.py env=cheetah_run num_train_steps=2e6 experiment=expert
python train.py env=walker_walk num_train_steps=2e6 experiment=expert
```

### Saving the expert demonstrations

Only needed if new expert demonstrations are needed. The parameter num_train_steps is set to be the same as when training the expert policy.

```
python save_expert_demonstration.py env=pendulum_swingup num_train_steps=1e6 experiment=expert_demonstration
python save_expert_demonstration.py env=cartpole_swingup num_train_steps=5e5 experiment=expert_demonstration
python save_expert_demonstration.py env=cheetah_run num_train_steps=2e6 experiment=expert_demonstration
python save_expert_demonstration.py env=walker_walk num_train_steps=2e6 experiment=expert_demonstration
```

### Training the imitation policies

The parameter num_train_steps is set to be the same as when training the expert policy in the agent environment.
```
python imitate.py env_expert=pendulum_swingup env_agent=cartpole_swingup num_train_steps=1e6 experiment=imitation_normalize gw_entropic=false gw_normalize=true
python imitate.py env_expert=cheetah_run env_agent=walker_walk num_train_steps=2e6 experiment=imitation_normalize gw_entropic=false gw_normalize=true
python imitate.py env_expert=MazeEnd_PointMass env_agent=MazeEnd_PointMass maze_id_expert=0 maze_id_agent=2 num_train_steps=1e6 experiment=imitation_normalize gw_entropic=false gw_normalize=true

```

## Credits

The code is based on the SAC Pytorch implementation available [here](https://github.com/denisyarats/pytorch_sac)

# Licensing
This repository is licensed under the
[CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
