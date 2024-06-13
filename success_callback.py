from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class SuccessRateCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SuccessRateCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.success_rates = []
        self.episode_successes = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Collect episode success information at the end of each rollout
        infos = self.locals["infos"]
        for info in infos:
            if 'is_success' in info:
                self.episode_successes.append(info['is_success'])

        self.episode_count += 1

        if self.episode_count % self.check_freq == 0:
            success_rate = np.mean(self.episode_successes)
            # self.logger.record('success_rate', success_rate)
            self.success_rates.append(success_rate)
            self.episode_successes = []

    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, "success_rates.npy"), np.array(self.success_rates))
