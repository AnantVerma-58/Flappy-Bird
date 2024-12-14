from gymnasium.envs.registration import register

register(
    id="flappyBird/GridWorld-v0",
    entry_point="flappyBird.envs:GridWorldEnv",
)
