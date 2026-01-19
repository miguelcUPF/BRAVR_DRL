use alvr_common::{error, info, warn};
use rand::{distributions::Distribution, distributions::WeightedIndex, thread_rng};
use std::{collections::VecDeque, path::PathBuf};
use tch::{nn, nn::Module, nn::OptimizerConfig, Kind, Tensor};

const N_ACTIONS: i64 = 3; // -1 = decrease, 0 = hold, 1 = increase

#[derive(Clone, Debug)]
pub struct SarsaAgentConfig {
    // Learning Hyperparameters
    pub gamma: f32,       // Discount factor (e.g., 0.99)
    pub lr: f64,          // Learning rate (e.g., 3e-4)
    pub tau: f64,         // Polyak averaging factor for soft updates (e.g., 0.005)
    pub temperature: f64, // Boltzmann exploration temperature (e.g., 0.5)
    pub epsilon: f64,     // minimum exploration probability (e.g., 0.05)
    pub n_step: usize,    // N-step return window (e.g., 4)

    // Neural Network Architecture
    pub state_dim: i64,
    pub hidden_dim: i64,

    // Action Shielding
    pub action_shielding_enabled: bool,

    // Persistence
    pub model_path: PathBuf,
    pub load_model: bool,
    pub save_model: bool,
}

/// Represents a single step of experience stored in the N-step buffer
struct Transition {
    s: Tensor,  // State at time t
    a_idx: i64, // Action taken at time t
    r: f32,     // Reward received at time t+1 (result of action a_t)
}

pub struct SarsaAgent {
    pub device: tch::Device,

    // VarStores hold the weights
    pub vs: nn::VarStore,
    pub target_vs: nn::VarStore,

    // The Neural Networks
    pub net: nn::Sequential,        // Main Network (updated by gradients)
    pub target_net: nn::Sequential, // Target Network (updated via Polyak averaging)

    pub opt: nn::Optimizer,
    pub cfg: SarsaAgentConfig,

    // N-Step Buffer: A FIFO queue to store history for N-step returns
    buffer: VecDeque<Transition>,

    // Stores the state and action from the previous step (t-1)
    pub s_prev: Option<Tensor>,
    pub a_prev_idx: Option<i64>,
}

impl SarsaAgent {
    /// Initialize SARSA agent with Deep Neural Function Approximation
    /// using a Double network approach (main + target) for stability.
    pub fn new(cfg: SarsaAgentConfig) -> Self {
        let device = if tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        // 1. Setup Main Network
        let mut vs = nn::VarStore::new(device);
        let net = Self::build_net(&vs.root(), &cfg);

        // Load existing model if requested
        if cfg.load_model {
            if cfg.model_path.exists() {
                info!("SARSA: Loading model from {:?}", cfg.model_path);
                if let Err(e) = vs.load(&cfg.model_path) {
                    error!("SARSA: Failed to load model: {:?}", e);
                }
            } else {
                warn!(
                    "SARSA: Load enabled but file not found at {:?}",
                    cfg.model_path
                );
            }
        }

        // 2. Setup Target Network
        // The target network is structurally identical but has its own independent weights.
        let mut target_vs = nn::VarStore::new(device);
        let target_net = Self::build_net(&target_vs.root(), &cfg);

        // Initialize target weights to match main weights exactly
        target_vs
            .copy(&vs)
            .expect("Failed to copy main->target varstore");

        // 3. Setup Optimizer (Adam)
        let opt = nn::Adam::default()
            .build(&vs, cfg.lr)
            .expect("Failed to build optimizer");

        Self {
            device,
            vs,
            target_vs,
            net,
            target_net,
            opt,
            cfg,
            buffer: VecDeque::new(),
            s_prev: None,
            a_prev_idx: None,
        }
    }

    /// Construct the MLP (Multi-Layer Perceptron)
    fn build_net(p: &nn::Path, cfg: &SarsaAgentConfig) -> nn::Sequential {
        nn::seq()
            .add(nn::linear(
                p / "l1",
                cfg.state_dim,
                cfg.hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                p / "l2",
                cfg.hidden_dim,
                cfg.hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                p / "out",
                cfg.hidden_dim,
                N_ACTIONS,
                Default::default(),
            ))
    }

    /// Select action using Boltzmann (Softmax) exploration with masking and epsilon floor.
    /// Returns: (selected_idx, q_values, probabilities, entropy, is_greedy)
    pub fn select_action(
        &self,
        s_t: &Tensor,
        mask: &[bool],
    ) -> (i64, Vec<f32>, Vec<f32>, f32, bool) {
        let s = s_t.to_device(self.device);

        // Use 'no_grad' because action selection is not part of backprop
        tch::no_grad(|| {
            // Forward pass through Main Network
            let q_values = self.net.forward(&s); // Output shape: [1, 3]
            let q_vec: Vec<f32> = Vec::try_from(q_values.view([-1])).unwrap();

            // Boltzmann distribution calculation
            // P(a) = exp(Q(s,a) / T) / sum(exp(Q(s,a) / T))
            let temp = self.cfg.temperature.max(1e-6) as f32;
            let max_q = q_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp: Vec<f32> = q_vec.iter().map(|q| ((q - max_q) / temp).exp()).collect(); // subtract max for numerical stability

            // Apply mask (if an action is forbidden, set its probability weight to 0)
            for (i, is_allowed) in mask.iter().enumerate() {
                if !*is_allowed {
                    exp[i] = 0.0;
                }
            }

            // Sum probabilities
            let sum: f32 = exp.iter().sum();

            // Normalize to get probabilities with safety check (if sum == 0, return uniform distribution)
            let mut probs: Vec<f32> = if sum > 1e-9 {
                exp.iter().map(|v| v / sum).collect()
            } else {
                vec![1.0 / N_ACTIONS as f32; N_ACTIONS as usize]
            };

            // Ensure minimum exploration (among allowed actions)
            let valid_count = mask.iter().filter(|&&b| b).count().max(1) as f32;
            let epsilon = self.cfg.epsilon as f32;
            // P_final = (1 - eps) * P_boltzmann + eps * P_uniform
            let mut mixed_probs: Vec<f32> = Vec::new();
            for (i, p) in probs.iter().enumerate() {
                if mask[i] {
                    mixed_probs.push((1.0 - epsilon) * p + epsilon / valid_count);
                } else {
                    mixed_probs.push(0.0);
                }
            }
            probs = mixed_probs;

            // Sample action from the probability distribution
            let dist = WeightedIndex::new(&probs).unwrap();
            let idx = dist.sample(&mut thread_rng()) as i64;

            // Calculate Policy Entropy: H(pi) = - sum( p * ln(p) )
            // Useful to monitor exploration (high entropy = high exploration)
            let entropy = -probs
                .iter()
                .map(|p| if *p > 1e-8 { p * p.ln() } else { 0.0 })
                .sum::<f32>();

            // Identify the greedy action (argmax) for logging purposes
            let argmax_idx = q_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap() as i64;

            (idx, q_vec, probs, entropy, idx == argmax_idx)
        })
    }

    /// Performs N-Step Expected Deep SARSA update.
    ///
    /// Math:
    /// We want to update Q(s_{t-N}, a_{t-N}) towards the N-step target G.
    /// G = R_{t-N+1} + gamma*R_{t-N+2} + ... + gamma^N * V_expected(s_{t+1})
    ///
    /// Returns: TD Error (scalar)
    pub fn update(&mut self, r_t: f32, s_t: &Tensor, a_t_idx: i64) -> f32 {
        let mut td_error = 0.0;
        if let (Some(last_s), Some(last_a)) = (&self.s_prev, self.a_prev_idx) {
            // Push transition: (S_{t-1}, A_{t-1}, R_t)
            self.buffer.push_back(Transition {
                s: last_s.shallow_clone(),
                a_idx: last_a,
                r: r_t,
            });

            // If buffer is full enough (warmup), learn
            if self.buffer.len() >= self.cfg.n_step {
                td_error = self.learn_from_buffer(s_t);
            }
        }

        // 2. Update Internal History Latch
        self.s_prev = Some(s_t.shallow_clone());
        self.a_prev_idx = Some(a_t_idx);

        td_error
    }

    fn learn_from_buffer(&mut self, s_next: &Tensor) -> f32 {
        let oldest = self.buffer.pop_front().unwrap();
        let s_old = oldest.s.view([1, -1]).to_device(self.device);
        let a_old_idx = Tensor::from_slice(&[oldest.a_idx])
            .to_kind(Kind::Int64)
            .to_device(self.device)
            .view([1, 1]);

        // Calculate the N-Step Return (G)
        let mut g: f32 = oldest.r;
        let mut discount: f32 = 1.0;
        for trans in self.buffer.iter() {
            discount *= self.cfg.gamma;
            g += discount * trans.r;
        }

        // Bootstrap: Calculate V(s_{t+1})
        let s_bootstrap = s_next.view([1, -1]).to_device(self.device);
        let final_gamma = discount * self.cfg.gamma;
        let v_expected = tch::no_grad(|| {
            let q_target_vals = self.target_net.forward(&s_bootstrap);
            let q_main_vals = self.net.forward(&s_bootstrap);

            let q_main_vec: Vec<f32> = Vec::try_from(q_main_vals.view([-1])).unwrap();
            let temp = self.cfg.temperature.max(1e-6) as f32;
            let sum: f32 = q_main_vec.iter().map(|q| (q / temp).exp()).sum();
            let probs: Vec<f32> = q_main_vec.iter().map(|q| (q / temp).exp() / sum).collect();

            let q_target_vec: Vec<f32> = Vec::try_from(q_target_vals.view([-1])).unwrap();
            probs
                .iter()
                .zip(q_target_vec.iter())
                .map(|(p, q)| p * q)
                .sum::<f32>()
        });
        g += final_gamma * v_expected;

        // Compute Loss
        let target_val = Tensor::from(g).to_device(self.device);
        let q_all = self.net.forward(&s_old);
        let q_pred = q_all.gather(1, &a_old_idx, false);
        let loss = q_pred.smooth_l1_loss(&target_val, tch::Reduction::Mean, 1.0);
        let td_error = f32::try_from(&(&target_val - &q_pred).abs()).unwrap_or(0.0);

        // Backprop
        self.opt.zero_grad();
        loss.backward();
        self.manual_clip_grad_norm(1.0);
        self.opt.step();

        // Update Target
        self.soft_update_target();

        td_error
    }

    /// Manual Gradient Clipping (L2 Norm)
    /// Prevents the updates from becoming too large and destabilizing the network.
    fn manual_clip_grad_norm(&self, max_norm: f64) {
        let vs = &self.vs;
        tch::no_grad(|| {
            let variables = vs.trainable_variables();
            let mut total_norm_sq = 0f64;

            // Calculate global norm
            for var in &variables {
                let grad = var.grad();
                if grad.defined() {
                    let grad_norm: f64 = grad.norm().double_value(&[]);
                    total_norm_sq += grad_norm * grad_norm;
                }
            }

            let total_norm = total_norm_sq.sqrt();
            let clip_coef = (max_norm / (total_norm + 1e-6)).min(1.0);

            // Apply clipping if necessary
            if clip_coef < 1.0 {
                for var in &variables {
                    let mut grad = var.grad();
                    if grad.defined() {
                        let _ = grad.f_mul_scalar_(clip_coef);
                    }
                }
            }
        });
    }

    /// Polyak Averaging for Target Network
    /// Moves target weights slowly towards main weights.
    /// theta_target = (1 - tau) * theta_target + tau * theta_main
    fn soft_update_target(&mut self) {
        let tau = self.cfg.tau;
        tch::no_grad(|| {
            let main_vars = self.vs.variables();
            let mut target_vars = self.target_vs.variables();

            for (name, tgt) in target_vars.iter_mut() {
                if let Some(main) = main_vars.get(name) {
                    // In-place update: tgt = tgt * (1 - tau)
                    let _ = tgt.f_mul_scalar_(1.0 - tau);
                    // In-place add: tgt = tgt + (main * tau)
                    let _ = tgt.f_add_(&(main * tau));
                }
            }
        });
    }

    /// Saves the Main Network weights to disk
    pub fn save_to_disk(&self) {
        if !self.cfg.save_model {
            return;
        }

        // Ensure directory structure exists
        if let Some(parent) = self.cfg.model_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match self.vs.save(&self.cfg.model_path) {
            Ok(_) => info!("SARSA: Saved model to {:?}", self.cfg.model_path),
            Err(e) => warn!("SARSA: Failed to save model: {:?}", e),
        }
    }
}
