# reach-ball-env

This repository contains a custom OpenAI Gym environment for a three-wheel car simulation using PyBullet. The environment is designed to train an agent to reach a soccer ball using the SAC algorithm in Stable Baselines3. The soccer ball and the car are located randomly in each episode.

## Demo Video

https://github.com/mininiii/reach-ball-env/assets/96100666/4b733cc5-aa53-43d3-98e0-f8c9743e1649

## Train

To start training the agent, run:

```
python3 sac_agent.py
```

If you want to visualize the training process, add the `--gui True` option:

```
python3 sac_agent.py --gui True
```

## Tensorboard
```
tensorboard --logdir=./logs
```
<img width="1096" alt="tensorboard_success_rate" src="https://github.com/mininiii/reach-ball-env/assets/96100666/538f92d1-f1ea-4599-aef0-9645520a1f40">




## Visualize Trained Model

To visualize the trained model, run:

```
python3 visualize.py
```

If you want to change the model path and the number of episodes, add the `--model_path {your_path} --len 5` option:
```
python3 visualize.py --model_path example/path/model --len 3
```


