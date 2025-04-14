from pendulum_env import InvertedPendulumEnv
env = InvertedPendulumEnv(render=True)
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        break
env.close()
