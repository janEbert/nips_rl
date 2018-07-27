import os
import argparse
from ast import literal_eval

from model import build_model, Agent
from state import NormState
import lasagne
from environments import RunEnv2
import config
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy', dest='accuracy', action='store', default=5e-5, type=float)
    parser.add_argument('--modeldim', dest='modeldim', action='store', default='3D',
            choices=('3d', '2d', '3D', '2D'), type=str)
    parser.add_argument('--prosthetic', dest='prosthetic', action='store', default=True, type=bool)
    parser.add_argument('--difficulty', dest='difficulty', action='store', default=0, type=int)
    parser.add_argument('--episodes', type=int, default=10, help="Number of test episodes.")
    parser.add_argument('--critic_layers', type=str, default='(64,32)', help="critic hidden layer sizes as tuple")
    parser.add_argument('--actor_layers', type=str, default='(64,64)', help="actor hidden layer sizes as tuple")
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normalization.")
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    args = parser.parse_args()
    args.modeldim = args.modeldim.upper()
    return args

def test_agent(args, testing, state_transform, num_test_episodes,
               model_params, weights, global_step, save_dir):
    env = RunEnv2(state_transform, visualize=True, integrator_accuracy=args.accuracy,
                  model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty,
                  skip_frame=1)
    test_rewards = []

    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
            build_model(**model_params)
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.set_actor_weights(weights)
    if args.weights is not None:
        actor.load(args.weights)

    for ep in range(num_test_episodes):
        seed = random.randrange(2**32-2)
        state = env.reset(seed=seed, difficulty=2)
        test_reward = 0
        while True:
            state = np.asarray(state, dtype='float32')
            action = actor.act(state)
            state, reward, terminal, _ = env._step(action)
            test_reward += reward
            if terminal:
                break
        test_rewards.append(test_reward)
    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)

    test_str ='global step {}; test reward mean: {:.2f}, std: {:.2f}, all: {} '.\
        format(global_step.value, float(mean_reward), float(std_reward), test_rewards)

    print(test_str)
    with open(os.path.join('test_report.log'), 'a') as f:
        f.write(test_str + '\n')

def main()
    args = get_args()
    args.critic_layers = literal_eval(args.critic_layers)
    args.actor_layers = literal_eval(args.actor_layers)

    save_dir = os.path.join('tests')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.move(save_dir, save_dir + '.backup')
        os.makedirs(save_dir)

    state_transform = NormState(args.prosthetic)
    # state_transform = StateVelCentr(obstacles_mode='standard',
    #                                 exclude_centr=True,
    #                                 vel_states=[])
    env = RunEnv2(state_transform, integrator_accuracy=args.accuracy, model=args.modeldim,
                  prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=1)
    env.change_model(args.modeldim, args.prosthetic, args.difficulty)
    num_actions = env.get_action_space_size()
    del env

    model_params = {
        'state_size': state_transform.state_size,
        'num_act': num_actions,
        'gamma': 0,
        'actor_layers': args.actor_layers,
        'critic_layers': args.critic_layers,
        'actor_lr': 0,
        'critic_lr': 0,
        'layer_norm': args.layer_norm
    }
    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(**model_params)
    actor = Agent(actor_fn, params_actor, params_crit)

    actor.load(args.weights)

    weights = [p.get_value() for p in params_actor]

    global_step = 0
    test_agent(args, testing, state_transform, args.episodes,
            model_params, _weights, global_step, save_dir)

if __name__ == '__main__':
    main()

