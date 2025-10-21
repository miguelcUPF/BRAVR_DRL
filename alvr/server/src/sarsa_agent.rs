use rand::Rng;
use tch::{nn, nn::Module, Tensor, nn::OptimizerConfig};

#[derive(Clone, Debug)]
pub struct SarsaAgentConfig {
    pub epsilon: f32,
    pub gamma: f32,
    pub lr: f64,
    pub state_dim: i64,
    pub hidden_dim: i64,
    pub action_values: Vec<f32>, // e.g., [-0.25, -0.1, 0.0, +0.1, +0.25]

    pub max_bitrate_mbps: f32,
    pub min_bitrate_mbps: f32,
    pub nfr_thresh: f32,
    pub rtt_thresh_ms: f32,
}

/// Neural network-based SARSA agent
pub struct SarsaAgent {
    pub vs: nn::VarStore,
    pub net: nn::Sequential,
    pub opt: nn::Optimizer,
    pub cfg: SarsaAgentConfig,

    // Internal state tracking for online updates
    pub s_prev: Option<Tensor>,
    pub a_prev_idx: Option<i64>,
}

impl SarsaAgent {
    // Initialize SARSA agent with neural function approximation
    // Q(s, a) is approximated by a neural network parameterized by θ
    pub fn new(cfg: SarsaAgentConfig) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);

        // Network architecture: Input (|S| state features) → Hidden → Output (|A| actions)
        // Each hidden layer has ReLU activation to ensure non-linearity
        // Each output neuron represents Q(s, a_i)
        let net = nn::seq()
            .add(nn::linear(
                &vs.root() / "l1",
                cfg.state_dim,
                cfg.hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                &vs.root() / "l2",
                cfg.hidden_dim,
                cfg.action_values.len() as i64,
                Default::default(),
            ));

        // Adam optimizer for parameter updates (θ ← θ - α∇L), minimizing loss L (MSE between Q(s, a) and Q'(s, a))
        let opt = nn::Adam::default().build(&vs, cfg.lr).expect("Failed to build optimizer");

        Self {
            vs,
            net,
            opt,
            cfg,
            s_prev: None,
            a_prev_idx: None,
        }
    }

    // ε-greedy action selection from Q(s,·)
    // - With probability ε: random exploration
    // - Otherwise: greedy exploitation
    /// Returns (action_value, action_index, is_greedy)
    pub fn select_action(&self, s_t: &Tensor) -> (f32, i64, bool) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.cfg.epsilon {
            // Random action (exploration)
            let idx = rng.gen_range(0..self.cfg.action_values.len());
            (self.cfg.action_values[idx], idx as i64, false)
        } else {
            // Greedy action (exploitation)
            let q_vals = self.net.forward(s_t);
            let idx = q_vals.argmax(1, false).int64_value(&[0]);
            (self.cfg.action_values[idx as usize], idx, true)
        }
    }

    // SARSA(0) online update
    //  Q(s_t, a_t) <- Q(s_t, a_t) + lr * (r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))
    pub fn update(&mut self, s_t: &Tensor, a_t_idx: i64, r_t: f32, s_tp1: &Tensor, a_tp1_idx: i64) {
        // Q(s_t, a_t)
        let q_pred = self.net.forward(s_t).gather(
            1,
            &Tensor::from_slice(&[a_t_idx as i64]).unsqueeze(0),
            false,
        );

        // Q(s_{t+1}, a_{t+1})
        let q_next = self.net.forward(s_tp1).gather(
            1,
            &Tensor::from_slice(&[a_tp1_idx as i64]).unsqueeze(0),
            false,
        );

        // Target: r + γ * Q(s_{t+1}, a_{t+1})
        let target = Tensor::from(r_t) + self.cfg.gamma * q_next.detach();

        // Loss = MSE(Q_pred, target) = [Q(s_t, a_t) - target]^2
        let loss = (q_pred - target)
            .pow_tensor_scalar(2.0)
            .mean(tch::Kind::Float);

        // Gradient descent update on θ using Adam optimizer
        self.opt.backward_step(&loss);
    }
}
