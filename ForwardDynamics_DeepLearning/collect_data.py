import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os

np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)
    X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    Y = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))
    step = 0.005
    torque_range = list(np.arange(-1.7, 1.7005, step))
    for i in torque_range:
        print(i)
        initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  
        initial_state[0] = -np.pi / 2.0
        arm_teacher.set_state(initial_state)
        action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
        action[0] = i
        arm_teacher.set_action(action)
        for j in range(900):
            X = np.hstack((X,np.vstack((arm_teacher.get_state(),action))))
            #print(X)
            arm_teacher.advance()
            Y = np.hstack((Y,arm_teacher.get_state()))
            #print(Y)
            
            
            
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
