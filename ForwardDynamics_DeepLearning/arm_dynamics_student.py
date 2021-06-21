from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from train_dynamics import Net


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = Net(9).to(device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        # ---
        self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            
            new_state = self.model.predict(np.vstack((state,action)))
            return new_state
            # ---
        else:
            return state
