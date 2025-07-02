from typing import Dict, List

class RewardScaler:
    """A buffer for storing rewards, online estimation of rewards standard deviations, and adaptively adjusting scale to match target std."""

    def __init__(self, target_ratios: Dict[str, float]):

        # First assert target ratios are positive and normalize
        assert all([ratio > 0 for ratio in target_ratios.values()])
        self.target_ratios = {name: ratio / sum(target_ratios.values()) for name, ratio in target_ratios.items()}

        self.reward_names = list(target_ratios.keys())

        self.rews: Dict[str, List[float]] = {name: [] for name in self.reward_names}
        self.mean: Dict[str, float] = {name: 0 for name in self.reward_names}
        self.mean_sq: Dict[str, float] = {name: 0 for name in self.reward_names}
        self.alpha = 0.01


    def _calculate_scale_factors(self) -> Dict[str, float]:
        stds = {name: (self.mean_sq[name] - self.mean[name]**2)**0.5 \
                    for name in self.reward_names}
        std_total = sum(stds.values())
        ratio = {name: stds[name] / std_total if std_total > 0 else self.target_ratios[name] \
                         for name in self.reward_names}
        print(f'current_ratio: {ratio}')
        scale_factor = {name: self.target_ratios[name] / ratio[name] \
                        if ratio[name] > 0 else 1.0 for name in self.reward_names}
        return scale_factor

    def get_scaled_reward(self, new_rewards: Dict[str, float]) -> float:
        assert len(new_rewards.keys()) == len(self.reward_names)

        scale_factor = self._calculate_scale_factors()
        print(f'scale_factor: {scale_factor}')
        scaled_rewards = {name: new_rewards[name] * scale_factor[name] for name in self.reward_names}

        return sum(scaled_rewards.values())
    
    # def _online_exponential_std(self, alpha: float = 0.05) -> None:
    #     for name in self.reward_names:
    #         if self.rews[name]:
    #             self.rew_running_avg[name] = alpha * self.rews[name][-1] + (1 - alpha) * self.rew_running_avg[name]
    #             self.rew_sq_running_avg[name] = alpha * self.rews[name][-1]**2 + (1 - alpha) * self.rew_sq_running_avg[name]

    def add_rewards(self, rewards: Dict[str, float]) -> None:
        for name, reward in zip(self.reward_names, rewards):
            # self.rews[name].append(reward)
            if self.rews[name]:
                self.mean[name] = self._exp_mean(reward, self.mean[name])
                self.mean_sq[name] = self._exp_mean(reward**2, self.mean_sq[name])
        # self._online_exponential_std()

    def _exp_mean(self, new: float, old: float) -> float:
        return self.alpha * new + (1 - self.alpha) * old

if __name__ == "__main__":
    reward_names = ['rew_abs_E', 'rew_valid', 'rew_basin']
    target_ratios = [0.5, 0.3, 0.2]

    reward_buffer = RewardScaler(dict(zip(reward_names, target_ratios)))

    new_rewards = [0.2, -0.3, 0.5]
    reward = reward_buffer.get_scaled_reward(new_rewards=dict(zip(reward_names, new_rewards)))
    reward_buffer.add_rewards(new_rewards)
    print(f'reward: {reward}')
