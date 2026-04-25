import gymnasium as gym

env = gym.make("MountainCar-v0")
state, _ = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, truncated, _ = env.step(action)
    print(state, reward)

for episode in range(5):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
print(total_reward)


