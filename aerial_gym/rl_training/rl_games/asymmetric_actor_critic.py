"""
Custom asymmetric actor-critic network for rl_games.
Actor sees first N dimensions, critic sees all N+M dimensions.
"""

import torch
import torch.nn as nn
from rl_games.algos_torch import network_builder


class AsymmetricActorCritic(network_builder.A2CBuilder.Network):
    """
    Asymmetric Actor-Critic network where:
    - Actor processes first `actor_obs_dim` dimensions
    - Critic processes all `actor_obs_dim + privileged_obs_dim` dimensions
    
    This enables asymmetric training where the critic has access to privileged information
    (e.g., ground truth target position) while the actor only uses observations available
    on the real robot (e.g., camera-based detections).
    """
    
    def __init__(self, params, **kwargs):
        # Get dimensions for asymmetric setup BEFORE parent init
        self.full_obs_dim = kwargs['input_shape'][0]  # Total input size (81 + 4 = 85)
        self.actor_obs_dim = params.get('actor_obs_dim', self.full_obs_dim)
        self.privileged_obs_dim = self.full_obs_dim - self.actor_obs_dim
        
        # Temporarily modify input_shape to actor_obs_dim for parent init
        original_input_shape = kwargs['input_shape']
        kwargs['input_shape'] = (self.actor_obs_dim,)
        
        # Initialize parent - this builds actor networks with actor_obs_dim
        super().__init__(params, **kwargs)
        
        # Restore original input shape
        kwargs['input_shape'] = original_input_shape
        
        print(f"\nAsymmetric Actor-Critic Network Configuration:")
        print(f"  Full observation dim: {self.full_obs_dim}")
        print(f"  Actor observation dim: {self.actor_obs_dim}")
        print(f"  Privileged obs dim: {self.privileged_obs_dim}")
        print(f"  Separate networks: {self.separate}")
        
        if not self.separate:
            raise ValueError("Asymmetric actor-critic requires separate=True in config!")
        
        # Rebuild critic MLP with full observation size (85D)
        # The parent built it with 81D, but critic needs 85D
        critic_mlp_args = {
            'input_size': self.full_obs_dim,  # Full size for critic!
            'units': self.units, 
            'activation': self.activation, 
            'norm_func_name': self.normalization,
            'dense_func': torch.nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer
        }
        self.critic_mlp = self._build_mlp(**critic_mlp_args)
        print(f"  → Rebuilt critic MLP with {self.full_obs_dim}D input")
        
    def forward(self, obs_dict):
        """
        Forward pass with asymmetric observations.
        
        Args:
            obs_dict: Dictionary with 'obs' key containing full observations [B, 85]
        
        Returns:
            mu, sigma, value, states (for continuous action space)
        """
        full_obs = obs_dict['obs']  # [B, 85]
        states = obs_dict.get('rnn_states', None)
        dones = obs_dict.get('dones', None)
        bptt_len = obs_dict.get('bptt_len', 0)
        
        # Split observations for actor and critic
        actor_obs = full_obs[:, :self.actor_obs_dim]  # [B, 81] - first 81 dims for actor
        critic_obs = full_obs  # [B, 85] - all 85 dims for critic
        
        # Process CNN if present
        if self.has_cnn:
            if self.permute_input and len(actor_obs.shape) == 4:
                actor_obs = actor_obs.permute((0, 3, 1, 2))
                critic_obs = critic_obs.permute((0, 3, 1, 2))
        
        # Forward through separate networks (must have separate=True)
        assert self.separate, "Asymmetric actor-critic requires separate=True in config!"
        
        # Process actor path
        a_out = actor_obs
        if self.has_cnn:
            a_out = self.actor_cnn(a_out)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
        
        # Process critic path
        c_out = critic_obs
        if self.has_cnn:
            c_out = self.critic_cnn(c_out)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
        
        # Debug: Print shapes on first forward pass
        if not hasattr(self, '_shapes_printed'):
            print(f"\n=== Forward Pass Verification ===")
            print(f"Actor obs shape: {actor_obs.shape}")
            print(f"Critic obs shape: {critic_obs.shape}")
            print(f"Actor MLP input shape: {a_out.shape}")
            print(f"Critic MLP input shape: {c_out.shape}")
            self._shapes_printed = True
        
        # Handle RNN
        if self.has_rnn:
            seq_length = obs_dict.get('seq_length', 1)
            
            if not self.is_rnn_before_mlp:
                a_out_in = a_out
                c_out_in = c_out
                
                # Verify dimensions before MLP
                if not hasattr(self, '_mlp_dims_printed'):
                    print(f"Actor MLP processing: input={a_out_in.shape}, expects {self.actor_obs_dim}D")
                    print(f"Critic MLP processing: input={c_out_in.shape}, expects {self.full_obs_dim}D")
                
                a_out = self.actor_mlp(a_out_in)
                c_out = self.critic_mlp(c_out_in)
                
                # Verify dimensions after MLP
                if not hasattr(self, '_mlp_dims_printed'):
                    print(f"After MLP - Actor: {a_out.shape}, Critic: {c_out.shape}")
                    print(f"=== Asymmetric actor-critic is working! ===\n")
                    self._mlp_dims_printed = True
                
                if self.rnn_concat_input:
                    a_out = torch.cat([a_out, a_out_in], dim=1)
                    c_out = torch.cat([c_out, c_out_in], dim=1)
            
            batch_size = a_out.size()[0]
            num_seqs = batch_size // seq_length
            a_out = a_out.reshape(num_seqs, seq_length, -1)
            c_out = c_out.reshape(num_seqs, seq_length, -1)
            
            if len(states) == 2:
                a_states = states[0]
                c_states = states[1]
            else:
                a_states = states
                c_states = states
            
            a_out = a_out.transpose(0, 1)
            c_out = c_out.transpose(0, 1)
            
            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1)
                dones = dones.transpose(0, 1)
            
            a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
            c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)
            
            a_out = a_out.transpose(0, 1)
            c_out = c_out.transpose(0, 1)
            
            a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
            c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)
            
            if self.rnn_ln:
                a_out = self.a_layer_norm(a_out)
                c_out = self.c_layer_norm(c_out)
            
            if type(a_states) is not tuple:
                a_states = (a_states,)
                c_states = (c_states,)
            states = a_states + c_states
            
            if self.is_rnn_before_mlp:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)
        else:
            a_out = self.actor_mlp(a_out)
            c_out = self.critic_mlp(c_out)
        
        # Compute value from critic path
        value = self.value_act(self.value(c_out))
        
        # Compute action from actor path
        if self.is_continuous:
            mu = self.mu_act(self.mu(a_out))
            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))
            return mu, sigma, value, states
        elif self.is_discrete:
            logits = self.logits(a_out)
            return logits, value, states
        elif self.is_multi_discrete:
            logits = [logit(a_out) for logit in self.logits]
            return logits, value, states


class AsymmetricBuilder(network_builder.A2CBuilder):
    """Custom builder that creates asymmetric actor-critic networks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, name, **kwargs):
        """Build asymmetric actor-critic network."""
        net = AsymmetricActorCritic(self.params, **kwargs)
        return net


def register_asymmetric_network():
    """Register the asymmetric actor-critic network with rl_games."""
    from rl_games.algos_torch import model_builder
    
    # Register the custom network builder
    model_builder.register_network('asymmetric_actor_critic', AsymmetricBuilder)
    print("Registered 'asymmetric_actor_critic' network builder")
