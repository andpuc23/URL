"""Environments' file"""
import numpy as np
from agent import Agent
from camera import Camera


class Environment:
    """Environment class"""

    def __init__(self,
                 points_coords: np.ndarray,
                 agent: Agent,
                 camera: Camera,
                 episode_length:int,
                 ) -> None:
        self.points_coords = points_coords
        self.agent = agent
        self.camera = camera
        self.episode_length = episode_length

    def get_observation(self) -> np.ndarray:
        """
        returns points' coordinates on camera image
        :return: array of points': [(i1, j1), (i2, j2), ... (i6, j6)]
        """
        return self.camera.project_points(self.points_coords)

    def running_cost(self) -> float:
        """
        returns sum of distances from true points to target values
        
        :return: float distance
        """
        return np.sqrt(
            np.power(
                self.get_observation() - self.agent.get_target_points(), 2
                ).sum(1)).sum()

    def play_episode(self) -> None:
        """
        resets the environment, performs :episode_length: steps
        """
        total_cost = 0.
        observations = []
        actions = []
        costs = []
        self.agent.set_initial_position()
        for ep in range(self.episode_length):
            observation = self.get_observation()
            self.agent.step(observation)
            cost = self.running_cost()
            total_cost += cost

            observations.append(observation)
            actions.append(action)
            costs.append(cost)
        self.agent.finetune(observations, actions, costs)






if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
