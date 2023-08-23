import pytest
from stable_baselines3 import A2C, PPO, DQN, HerReplayBuffer
from sb3-contrib import QRDQN, TRPO, ARS, RPPO

def test_qrdqn():
    model = QRDQN(
        "MlpPolicy",
        "LuxAI_S2-v0",
        policy_kwargs=dict(n_quantiles=25, net_arch=[128, 128]),
        learning_starts=100,
        buffer_size=800,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=500)

def test_qrdqn_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = QRDQN(
        "MlpPolicy",
        "LuxAI_S2-v0",
        policy_kwargs=dict(n_quantiles=25, net_arch=[128, 128]),
        learning_starts=100,
        buffer_size=800,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./qrdqn_tensorboard/",
    )
    model.learn(total_timesteps=500, callback=eval_callback)

def test_trpo():
    model = TRPO(
        "MlpPolicy",
        "LuxAI_S2-v0",
        n_steps=128,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1)
    model.learn(total_timesteps=500)

def test_trpo_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = TRPO(
        "MlpPolicy",
        "LuxAI_S2-v0",
        n_steps=128,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
        tensorboard_log="./trpo_tensorboard/",
    )
    model.learn(total_timesteps=500, callback=eval_callback)

def test_ars():
    model = ARS(
        "MlpPolicy",
        "LuxAI_S2-v0",
        n_delta=1,
        verbose=1,
        seed=0)
    model.learn(total_timesteps=500, log_interval=1)

def test_ars_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = ARS(
        "MlpPolicy",
        "LuxAI_S2-v0",
        n_delta=1,
        verbose=1,
        seed=0,
        tensorboard_log="./ars_tensorboard/",
    )
    model.learn(total_timesteps=500, callback=eval_callback)

@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {},
        dict(shared_lstm=True, enable_critic_lstm=False),
        dict(
            enable_critic_lstm=True,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
        dict(
            enable_critic_lstm=False,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
    ],
)
def test_policy_kwargs(policy_kwargs):
    model = RPPO(
        "MlpLstmPolicy",
        "LuxAI_S2-v0",
        n_steps=16,
        seed=0,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=32)

def test_cnn_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = RPPO(
        "MlpLstmPolicy",
        "LuxAI_S2-v0",
        n_steps=16,
        seed=0,
        policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)),
        n_epochs=2,
        tensorboard_log="./cnn_tensorboard/",
    )

    model.learn(total_timesteps=32, callback=eval_callback)


def test_a2c():
    model = A2C(
        "MlpPolicy",
        "LuxAI_S2-v0",
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1)
    model.learn(total_timesteps=64)

def test_a2c_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = A2C(
        "MlpPolicy",
        "LuxAI_S2-v0",
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
        tensorboard_log="./a2c_tensorboard/",
    )
    model.learn(total_timesteps=64, callback=eval_callback)


def test_ppo():
    
    model = PPO(
        "MlpPolicy",
        "LuxAI_S2-v0",
        n_steps=512,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
        n_epochs=2,
        )
    model.learn(total_timesteps=1000)

def test_ppo_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = PPO(
        "MlpPolicy",
        "LuxAI_S2-v0",
        n_steps=512,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
        n_epochs=2,
        tensorboard_log="./ppo_tensorboard/",
    )
    model.learn(total_timesteps=1000, callback=eval_callback)


def test_dqn():
    model = DQN(
        "MlpPolicy",
        "LuxAI_S2-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=200)

def test_dqn_with_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env="LuxAI_S2-v0",
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = DQN(
        "MlpPolicy",
        "LuxAI_S2-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
    )
    model.learn(total_timesteps=200, callback=eval_callback)


def test_performance_her():

    model = DQN(
        "MultiInputPolicy",
        "LuxAI_S2-v0",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=5,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=5e-4,
        train_freq=1,
        gradient_steps=1,
        learning_starts=100,
        exploration_final_eps=0.02,
        target_update_interval=500,
        seed=0,
        batch_size=32,
        buffer_size=int(1e5),
    )

    model.learn(total_timesteps=5000, log_interval=50)

def test_performance_her_with_callback():
    
        from stable_baselines3.common.callbacks import EvalCallback
    
        eval_callback = EvalCallback(
            eval_env="LuxAI_S2-v0",
            eval_freq=100,
            deterministic=True,
            render=False,
            verbose=1,
        )
    
        model = DQN(
            "MultiInputPolicy",
            "LuxAI_S2-v0",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=5,
                goal_selection_strategy="future",
            ),
            verbose=1,
            learning_rate=5e-4,
            train_freq=1,
            gradient_steps=1,
            learning_starts=100,
            exploration_final_eps=0.02,
            target_update_interval=500,
            seed=0,
            batch_size=32,
            buffer_size=int(1e5),
            tensorboard_log="./her_tensorboard/",
        )
    
        model.learn(total_timesteps=5000, log_interval=50, callback=eval_callback)