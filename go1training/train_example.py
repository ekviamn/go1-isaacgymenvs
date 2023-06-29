import gym
from tqdm import trange
from params_proto import ParamsProto


class Args(ParamsProto):
    seed = 100
    # num_envs = 40_000
    # num_eval_envs = 10_000
    num_steps = 50_000
    eval_freq = 10_000
    eval_steps = 1000

def eval_fn(eval_env, policy, eval_steps, filename=f"videos/agent.mp4"):
    from ml_logger import logger

    obs = eval_env.reset()
    # policy.eval()
    frames = []
    for i in trange(eval_steps + 1):

        act = policy(obs)
        obs, reward, done, info = eval_env.step(act)
        logger.store_metrics(reward=reward)
        img = eval_env.render("rgb_array")
        frames.append(img)
        if done:
            obs = eval_env.reset()

    logger.save_video(frames, filename, fps=24)

def train(**deps):
    from ml_logger import logger

    Args._update(deps)
    logger.log_params(Args=vars(Args))

    # Set up environment
    env = gym.make("Acrobot-v1")
    eval_env = gym.make("Acrobot-v1")

    policy = lambda x: env.action_space.sample()

    obs = env.reset()
    pbar = trange(Args.num_steps + 1)
    for step in pbar:

        # sample
        act = policy(obs)
        obs, reward, done, info = env.step(act)

        if done:
            obs = env.reset()

        if step % Args.eval_freq == 0:
            eval_fn(eval_env, policy, eval_steps=Args.eval_steps, filename=f"videos/{step:06d}")
            logger.log_metrics_summary(key_values=dict(step=step))


if __name__ == '__main__':
    logger.prefix = "/s"
    logger.log_text("""
    charts:
    - yKey: "reward"
      xKey: "step"
    """, ".charts.yml", True, True)
    train()

    # thunk = instr(train)
    # thunk()