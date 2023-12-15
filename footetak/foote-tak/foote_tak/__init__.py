from gym.envs.registration import register

register(
    id="Tak-v0",
    entry_point="foote_tak.envs:TakEnv",
    kwargs={
        'board_size': 3,
        'pieces': 10
    }
)


# register(
#     id="Tak-v0",
#     entry_point="foote_tak.envs:TakEnv",
#     kwargs={
#         'board_size': 3,
#         'pieces': 10
#     }
# )