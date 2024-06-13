
import os
import torch
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from carEnv import ThreeWheelCarEnv
from callback import SaveOnBestTrainingRewardCallback
import argparse
from success_callback import SuccessRateCallback

def train(env_id='ThreeWheelCar', gui_mode=False, algo_class=SAC, log_base_dir="logs", model_base_dir="models"):
    # set arguments --------------------------------------------------------------------------
    seed = 256
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    
    model_name = algo_class.__name__ + "_" + env_id
    model_path = os.path.join(model_base_dir, model_name)
    callback_log_path = os.path.join(model_base_dir, model_name, "temp")
    log_path = os.path.join(log_base_dir, model_name)


    # create the environment ------------------------------------------------------------------
    env = ThreeWheelCarEnv(gui=gui_mode)
    env = Monitor(env, os.path.join(callback_log_path, 'monitor.csv'))    
    
    # set the model : https://github.com/DLR-RM/rl-trained-agents/tree/master/her --------------
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        online_sampling=True,
        max_episode_length=200, 
    )
    

    if env_id == "ThreeWheelCar":
        total_iterations = 250_000
        learning_starts = 1000
        batch_size = 256
        policy_kwargs = {
            'net_arch': [64, 64],
            'n_critics': 1,        
        }
        
    model = algo_class(
        policy="MultiInputPolicy", 
        env=env, 
        buffer_size=50000,
        learning_starts=learning_starts,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        gamma=0.95,
        # --------------------------------        
        tau=0.005,        
        learning_rate=0.001,        
        train_freq=1,
        gradient_steps=1,     
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        # --------------------------------                   
        replay_buffer_class=HerReplayBuffer, 
        replay_buffer_kwargs=replay_buffer_kwargs,
        tensorboard_log=log_path,
        verbose=1,
        seed=seed,
        device='auto',                
    )

    # train the model -----------------------------------------------------------------------------
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        log_dir=callback_log_path
    )
    success_rate_callback = SuccessRateCallback(
        check_freq=1000, 
        log_dir=callback_log_path
    )
    model.learn(
        total_timesteps=total_iterations,
        callback=[callback, success_rate_callback],
        log_interval=10,
        tb_log_name="SAC",
        reset_num_timesteps=True,
        progress_bar=True,
    )


    # save the trained model ----------------------------------------------------------------------
    model.save(model_path+"/final_model")
    print("model saved to {}".format(model_path+"/final_model.zip"))
        
    # close the environment
    del model
    env.close()
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', type=bool, default=False, help='Enable GUI mode')
    args = parser.parse_args()
    gui_mode = args.gui
    
    env_id = "ThreeWheelCar"
    train(env_id, gui_mode)
