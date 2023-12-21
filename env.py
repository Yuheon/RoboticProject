import numpy as np
import matplotlib.pyplot as plt

class PathPlanningEnv:
    def __init__(self, level=0, agent_jump=0):
        self.grid_size = (10, 10)  # height, width
        
        if level == 0:
            self.obstacles = 10
        elif level == 1:
            self.obstacles = 20
        elif level == 2:
            self.obstacles = 30
        else:
            raise NotImplementedError(f"Environment level:{level} is not implemented.")
        
        self.agent_jump = agent_jump
        self.obstacle_height_range = (1, 3)
        self.start = (0, 0)
        self.goal = (self.grid_size[0] - 1 , self.grid_size[1] - 1)
        self.agent_position = self.start
        self.grid = self._make_grid()
        self.done = False
        self.step_reward = -0.1
        self.goal_reward = 10
        self.n_actions = 4
        

    def _make_grid(self):
        grid = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=int)
        
        # grid[self.start] = -1   # agent in starting point
        # grid[self.goal] = -2   # Goal

        # Randomly place obstacles with random heights
        count = 0
        while count < self.obstacles:
            x = np.random.randint(0, self.grid_size[1], size=1) 
            y = np.random.randint(0, self.grid_size[0], size=1) 
            if (x, y) != self.start and (x, y) != self.goal and grid[x, y] == 0:
                grid[x, y] = np.random.randint(self.obstacle_height_range[0], self.obstacle_height_range[1] + 1)
                count += 1
        
        grid = grid - self.agent_jump
        grid[np.where(grid < 0)] = 0  # agent jump > height of obstacle can be viewed as no obsatcle
        grid[np.where(grid > 0)] = 1
        grid[self.goal] = -2
        
        return grid

    def reset(self):
        self.agent_position = self.start
        self.grid = self._make_grid()
        copy_grid = self.grid.copy()
        copy_grid[self.start] = -1  # agent in starting point
        self.done = False
        # return (self.grid, self.agent_position), 0, False
        return copy_grid, 0, False  # state, reward, done

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Call reset() to start a new episode.")


        if action == 0:  # UP
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 2:  # Right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)
        elif action == 3:  # Down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        else:
            raise ValueError("Action must be in range [0, 1, 2, 3]")

        # Check if new position is valid
        # wall & obstacle
        if (0 <= new_position[0] < self.grid_size[0]) and \
            (0 <= new_position[1] < self.grid_size[1]) and \
                (self.grid[new_position] in [0, -2]):

            self.agent_position = new_position
        else:
            # Invalid move (obstacle or outside grid), stay in place
            new_position = self.agent_position

        # Check for goal
        reward = self.step_reward
        if new_position == self.goal:
            reward += self.goal_reward
            self.done = True

        # return (self.grid, new_position), reward, self.done
        copy_grid = self.grid.copy()
        copy_grid[self.agent_position] = -1
        return copy_grid, reward, self.done # state, reward, done

    def render(self):  # Render the environment
        env_grid = self.grid.copy()
        env_grid[self.agent_position] = -1
        
        plt.imshow(env_grid)
        plt.show()

if __name__ == "__main__":
    
    # environment configuration
    level=0
    agent_jump=0
    
    # make environment with env config
    env = PathPlanningEnv(level, agent_jump)
    state, reward, done = env.reset()  # Reset the environment before starting
    
    env.render()  # Initial render

    n_tries = 5  # the numboer of taken random actions

    # Example of agent taking random steps until it reaches the goal
    for i in range(n_tries):
        action = np.random.choice(4)  # Choose a random action
        state, reward, done = env.step(action)
        env.render()  # Render the environment after the step
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
        print(f"New state")
        print(f"grid:\n{state}")
        if done:
            break  # The goal is reached

    # Please note that this is a very basic example. In a real RL setup, you'd have an agent learning from this interaction.
