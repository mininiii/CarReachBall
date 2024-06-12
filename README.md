# CarReachBall

This repository contains a custom OpenAI Gym environment for a three-wheel car simulation using PyBullet. The environment is designed to train an agent to reach a soccer ball using the SAC algorithm in Stable Baselines3. The soccer ball and the car are located randomly in each episode.

## Demo Video



## Train

To start training the agent, run:

```
python3 sac_agent.py
```

If you want to visualize the training process, add the `--gui True` option:

```
python3 sac_agent.py --gui True
```

## Visualize Trained Model

To visualize the trained model, run:

```
python3 visualize.py
```
