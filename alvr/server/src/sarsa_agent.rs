use alvr_common::{error, info, warn};
use rand::{distributions::Distribution, distributions::WeightedIndex, thread_rng};
use std::path::PathBuf;
use tch::{nn, nn::Module, nn::OptimizerConfig, Kind, Tensor};

#[derive(Clone, Debug)]
pub struct SarsaAgentConfig {
    // Learning hyperparameters
    pub gamma: f32,
    pub lr: f64,
    pub tau: f64,         // polyak factor for soft updates
    pub temperature: f64, // for Boltzmann selection

    // Neural network architecture
    pub state_dim: i64,
    pub hidden_dim: i64,
    pub action_values: Vec<f32>, // e.g., [-0.25, -0.1, 0.0, +0.1, +0.25]

    // Normalization bounds
    pub max_bitrate_mbps: f32,
    pub min_bitrate_mbps: f32,

    // Reward knobs
    pub nfr_thresh: f32,
    pub rtt_target_ms: f32,
    pub rtt_tolerance_factor: f32, // must be > 1

    // Reward weights
    pub w_bitrate: f32,
    pub w_nfr: f32,
    pub w_rtt: f32,
    pub w_vol: f32,
    pub w_fairness: f32,

    // File loading/saving
    pub model_path: PathBuf,
    pub load_model: bool,
    pub save_model: bool,
}

/// Neural network-based SARSA agent
pub struct SarsaAgent {
    pub device: tch::Device,

    // Main and Target varstores
    pub vs: nn::VarStore,
    pub target_vs: nn::VarStore,

    // Main and Target neural networks
    pub net: nn::Sequential,
    pub target_net: nn::Sequential,

    pub opt: nn::Optimizer,
    pub cfg: SarsaAgentConfig,

    // for SARSA online updates
    pub s_prev: Option<Tensor>,
    pub a_prev_idx: Option<i64>,
}

impl SarsaAgent {
    // Initialize SARSA agent with Deep Neural Function Approximation using a Double network approach (main + target) for stability
    pub fn new(cfg: SarsaAgentConfig) -> Self {
        let device = if tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let mut vs = nn::VarStore::new(device);
        let net = Self::build_net(&vs.root(), &cfg);

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

        let mut target_vs = nn::VarStore::new(device);
        let target_net = Self::build_net(&target_vs.root(), &cfg);

        // copy weights from main to target
        target_vs
            .copy(&vs)
            .expect("Failed to copy main->target varstore");

        let opt = nn::Adam::default()
            .build(&vs, cfg.lr)
            .expect("Failed to build optimizer");

        Self {
            vs,
            target_vs,
            net,
            target_net,
            opt,
            cfg,
            s_prev: None,
            a_prev_idx: None,
            device,
        }
    }

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
                cfg.action_values.len() as i64,
                Default::default(),
            ))
    }

    // Boltzmann (softmax) action selection. Returns (action_value, action_idx, matches_argmax)
    pub fn select_action(&self, s_t: &Tensor) -> (f32, i64, bool) {
        // ensure the state tensor is on the agent device
        let s = s_t.to_device(self.device);
        let q_values = self.net.forward(&s); // shape [1, n_actions]

        // avoid degenerate temperature
        let temp = self.cfg.temperature.max(1e-6);
        let scaled = &q_values / temp;
        let probs = scaled.softmax(-1, Kind::Float);

        // sample
        let probs_vec: Vec<f32> = Vec::try_from(probs.view([-1])).expect("probs->vec");
        let dist = WeightedIndex::new(&probs_vec).expect("Invalid softmax probs");
        let mut rng = thread_rng();
        let idx = dist.sample(&mut rng) as i64;

        let argmax_idx = q_values.argmax(1, false).int64_value(&[0]);
        let matches_argmax = idx == argmax_idx;

        (self.cfg.action_values[idx as usize], idx, matches_argmax)
    }

    // Perform DEEP SARSA update step
    pub fn update(
        &mut self,
        s_t: &Tensor,
        a_t_idx: i64,
        r_t: f32,
        s_next: &Tensor,
        a_next_idx: i64,
    ) -> (f32, f32) {
        // Returns (Loss, Q_Predicted)
        let s = s_t.view([1, -1]).to_device(self.device);
        let s_n = s_next.view([1, -1]).to_device(self.device);

        // ensure index tensors are i64 and on correct device
        let idx_t = Tensor::from_slice(&[a_t_idx])
            .to_kind(Kind::Int64)
            .to_device(self.device)
            .view([1, 1]);

        let idx_n = Tensor::from_slice(&[a_next_idx])
            .to_kind(Kind::Int64)
            .to_device(self.device)
            .view([1, 1]);

        // Q_pred = Q_main(s, a)
        let q_all = self.net.forward(&s); // [1, n_actions]
        let q_pred = q_all.gather(1, &idx_t, false);

        // Q_next (target) using the target network; detach to stop gradients
        let q_next_all = self.target_net.forward(&s_n);
        let q_next = q_next_all.gather(1, &idx_n, false).detach();

        // Bellman equation for SARSA
        let target = Tensor::from(r_t).to_device(self.device) + (self.cfg.gamma as f32) * q_next;

        // Smooth L1 (Huber) loss for stability
        let loss = q_pred.smooth_l1_loss(&target, tch::Reduction::Mean, 1.0);

        let q_val_scalar = f32::try_from(q_pred).unwrap_or(0.0);
        let loss_scalar = f32::try_from(&loss).unwrap_or(0.0);

        // gradient descent step
        // self.opt.backward_step(&loss);
        self.opt.zero_grad(); // reset gradients
        loss.backward(); // compute gradients (backprop)
        self.manual_clip_grad_norm(1.0); // clip gradients to avoid explosion
        self.opt.step(); // update weights

        // soft-update target (polyak averaging)
        self.soft_update_target();

        (loss_scalar, q_val_scalar)
    }

    // Manual implementation of Gradient Clipping (L2 Norm)
    // This scales down the gradients of all variables if their combined magnitude exceeds max_norm.
    fn manual_clip_grad_norm(&self, max_norm: f64) {
        let vs = &self.vs;
        tch::no_grad(|| {
            let variables = vs.trainable_variables();

            // 1. Compute total grad norm (L2)
            let mut total_norm_sq = 0f64;

            for var in &variables {
                let grad = var.grad();
                if grad.defined() {
                    let grad_norm: f64 = grad.norm().double_value(&[]);

                    total_norm_sq += grad_norm * grad_norm;
                }
            }

            let total_norm = total_norm_sq.sqrt();
            let clip_coef = (max_norm / (total_norm + 1e-6)).min(1.0);

            // 2. Scale gradients in-place
            if clip_coef < 1.0 {
                for var in &variables {
                    let mut grad = var.grad();
                    if grad.defined() {
                        let _ = grad
                            .f_mul_scalar_(clip_coef)
                            .expect("Failed to scale gradient");
                    }
                }
            }
        });
    }

    /// Polyak averaging
    fn soft_update_target(&mut self) {
        let tau = self.cfg.tau;
        tch::no_grad(|| {
            // build hashmaps for quick lookup by name
            let main_vars = self.vs.variables();
            let mut target_vars = self.target_vs.variables();

            for (name, tgt) in target_vars.iter_mut() {
                if let Some(main) = main_vars.get(name) {
                    // tgt = (1 - tau) * tgt + tau * main
                    // do it in-place to avoid allocations
                    let _ = tgt.f_mul_scalar_(1.0 - tau).unwrap();
                    let _ = tgt.f_add_(&(main * tau)).unwrap();
                }
            }
        });
    }

    pub fn save_to_disk(&self) {
        if !self.cfg.save_model {
            return;
        }

        // Ensure parent folder exists
        if let Some(parent) = self.cfg.model_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match self.vs.save(&self.cfg.model_path) {
            Ok(_) => info!("SARSA: Saved model to {:?}", self.cfg.model_path),
            Err(e) => warn!("SARSA: Failed to save model: {:?}", e),
        }
    }
}
