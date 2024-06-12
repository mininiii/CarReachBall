from carEnv import ThreeWheelCarEnv
from stable_baselines3 import SAC
import argparse


def visualize(len, model_path):
    # 환경을 GUI 모드로 다시 초기화
    env = ThreeWheelCarEnv(gui=True)

    # 학습된 모델 로드
    if model_path[-4:] == ".zip":
        model_path = model_path[:-4]
        
    model = SAC.load(model_path, env=env)
    # model = SAC.load("models/SAC_ThreeWheelCar/final_model", env=env)

    for i in range(len):
        obs, done = env.reset(), False
        print("===================================")
        episode_reward = 0
        while not done:
            env.render()
            action = model.predict(obs)[0]
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        print("Episode reward", episode_reward)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/SAC_ThreeWheelCar_basic/temp/best_model", help='Path to the trained model')
    parser.add_argument('--len', type=int, default=10, help='Number of episodes to visualize')
    args = parser.parse_args()
    model_path = args.model_path
    
    visualize(model_path=model_path, len=args.len)
    
