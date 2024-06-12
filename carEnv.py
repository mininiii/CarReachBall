import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import random


class ThreeWheelCarEnv(gym.Env):
    
    def __init__(self, gui=False, max_episode_len=200):
        super(ThreeWheelCarEnv, self).__init__()

        # 상태 공간을 Dict로 정의
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        })
        
        self.action_space = spaces.Box(low=0, high=8, shape=(1,), dtype=np.int8)
        
        # PyBullet 초기화
        if gui:
            p.connect(p.GUI)
        else: 
            p.connect(p.DIRECT)

        # Set the initial position and orientation of the view in GUI
        p.resetDebugVisualizerCamera(
            cameraDistance=7.0,     # distance from eye to camera target position
            cameraYaw=90,            # camera yaw angle (in degrees) left/right
            cameraPitch=-70,        # camera pitch angle (in degrees) up/down
            cameraTargetPosition=[0.55, -0.35, 0.2]  #  the camera focus point
        )

        self.car_id = None
        self.ball_id = None
        self.step_counter = 0
        self.max_episode_len = max_episode_len
        self.gui = gui
        

    def reset(self):
        # Reset simulation and environment
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)
        p.setPhysicsEngineParameter(enableConeFriction=1, contactBreakingThreshold=0.001)
        self.step_counter = 0
        
        # 자동차와 공의 위치를 랜덤하게 설정
        self.car_position = [random.uniform(1, 3), random.uniform(-2, 2)]
        self.car_orientation = np.random.uniform(low=-np.pi, high=np.pi)
        self.ball_position = [random.uniform(-5, -1), random.uniform(-3, 3)]

        # 자동차와 공을 URDF 파일에서 로드
        urdf_path = pybullet_data.getDataPath()
        plane_path = os.path.join(urdf_path, "samurai.urdf")
        ball_path = os.path.join(urdf_path, "soccerball.urdf")

        p.loadURDF(plane_path)
        self.car_id = p.loadURDF("data/robot.xacro", 
                                 basePosition=[self.car_position[0], self.car_position[1], 0.05],
                                 baseOrientation=p.getQuaternionFromEuler([0, 0, self.car_orientation])) 
        
        self.ball_id = p.loadURDF(ball_path, basePosition=[self.ball_position[0], self.ball_position[1], 1], globalScaling=0.4)
        
        for i in range(100):
            p.stepSimulation()

        return self._get_obs()

    def _get_obs(self):
        car = p.getBasePositionAndOrientation(self.car_id)
        ball = p.getBasePositionAndOrientation(self.ball_id)
        car_pos, car_orn = car[0], car[1]
        ball_pos, ball_orn = ball[0], ball[1]
        invCarPos, invCarOrn = p.invertTransform(car_pos, car_orn)
        ballPosInCar, _ = p.multiplyTransforms(invCarPos, invCarOrn, ball_pos, ball_orn)

        observation = np.array([ballPosInCar[0], ballPosInCar[1]], dtype=np.float32)

        
        achieved_goal = np.array([car_pos[0], car_pos[1]], dtype=np.float32)
        desired_goal = np.array([ball_pos[0], ball_pos[1]], dtype=np.float32)
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        closestPoints = p.getClosestPoints(self.car_id, self.ball_id, 10000)

        numPt = len(closestPoints)
        reward = -1000
        if (numPt > 0):
            reward = -closestPoints[0][8]
            if closestPoints[0][8] < 0.3:
                reward = 200
        return reward

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]

        forward = fwd[int(action[0])]*0.3
        steer = steerings[int(action[0])]

    
        # 각 휠의 속도를 계산 (차량의 기하학적 특성에 따라 다를 수 있음)
        wheel_separation = 0.13  # 좌우 바퀴 사이의 거리 (미터)
        wheel_radius = 0.015  # 바퀴 반지름 (미터)

        left_wheel_velocity = (forward - (wheel_separation / 2.0) * steer) / wheel_radius
        right_wheel_velocity = (forward + (wheel_separation / 2.0) * steer) / wheel_radius

        # PyBullet에서 휠 속도 설정
        p.setJointMotorControl2(self.car_id, 2, p.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity, force=100)
        p.setJointMotorControl2(self.car_id, 3, p.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity, force=100)

        for _ in range(50):
            p.stepSimulation()
            self.step_counter += 1
        
        obs = self._get_obs()

        
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})

        
        done = reward == 200 or self.step_counter > 6000
        return obs, reward, done, {}

    def render(self, mode='human'):
        car_pos, _ = p.getBasePositionAndOrientation(self.car_id)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            # cameraTargetPosition=[0.7, 0.0, 0.05],
            cameraTargetPosition=car_pos,
            distance=5.5,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(960) / 720,
            nearVal=0.1,
            farVal=100.0
        )
        width, height, rgb_pixels, depth_pixels, _ = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(rgb_pixels, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.array(depth_pixels, dtype=np.float32)
        depth_array = np.reshape(depth_array, (height, width))
        depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

        if mode == 'human':
            return None
        elif mode == 'rgb_array':
            return rgb_array
        elif mode == 'depth_array':
            depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
            depth_image = (depth_norm * 255).astype(np.uint8)
            return depth_image
        else:
            raise ValueError("Unsupported mode. Supported modes are 'human', 'rgb_array', and 'depth_array'.")
        
    def close(self):
        if p.isConnected():
            p.disconnect()