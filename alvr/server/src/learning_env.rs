use tch::Tensor;

pub const STATE_DIM: i64 = 13;
const MAX_MCS: f32 = 11.0; // assuming 802.11ax

#[derive(Clone, Debug)]
pub struct LearningConfig {
    pub bitrate_levels_mbps: Vec<f32>, // The discrete bitrate ladder

    // Targets
    pub nfr_target: f32,
    pub rtt_target_ms: f32,

    // Scales
    pub rtt_state_scale_ms: f32, // Divisor for tanh, e.g., 100.0 (values over 100ms map to ~1.0)

    // Weights
    pub w_bitrate: f32,
    pub w_nfr: f32,
    pub w_rtt: f32,
    pub w_osc: f32,
    pub w_fairness: f32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            bitrate_levels_mbps: vec![2.0, 5.0, 10.0, 20.0, 50.0], // should be ordered
            nfr_target: 0.95,
            rtt_target_ms: 22.0,

            rtt_state_scale_ms: 100.0,

            w_bitrate: 1.0,
            w_nfr: 0.5,
            w_rtt: 3.0,
            w_osc: 0.05,
            w_fairness: 0.5,
        }
    }
}

impl LearningConfig {
    pub fn new(
        bitrate_levels_mbps: Vec<f32>,
        nfr_target: f32,
        rtt_target_ms: f32,
        rtt_state_scale_ms: f32,
        w_bitrate: f32,
        w_nfr: f32,
        w_rtt: f32,
        w_osc: f32,
        w_fairness: f32,
    ) -> Self {
        Self {
            bitrate_levels_mbps,
            nfr_target,
            rtt_target_ms,
            rtt_state_scale_ms,
            w_bitrate,
            w_nfr,
            w_rtt,
            w_osc,
            w_fairness,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EnvironmentSnapshot {
    pub nfr: f32,
    pub rtt_ms: f32,
    pub bitrate_bps: f32,
    pub bitrate_idx: usize,

    // AP Telemetry
    pub mcs_raw: f32,
    pub channel_busy_pct: f32,
    pub my_airtime_fraction: f32,
    pub active_vr_count: usize,
    pub fairness_index: f32,
}

pub struct StreamingEnvironment {
    pub cfg: LearningConfig,
    prev_norm_vals: Option<(f32, f32, f32)>, // (nfr, rtt, mcs) normalized
    prev_bitrate_idx: Option<usize>,         // The index at t-1
    last_move_dir: i32,                      // Direction of move (t-2 -> t-1)
}

impl StreamingEnvironment {
    pub fn new(cfg: LearningConfig) -> Self {
        Self {
            cfg,
            prev_norm_vals: None,
            prev_bitrate_idx: None,
            last_move_dir: 0,
        }
    }

    pub fn build_state_vector(&mut self, snap: &EnvironmentSnapshot) -> Tensor {
        // 1. Normalize Current Inputs
        let nfr_norm = snap.nfr.clamp(0.0, 1.0);
        let rtt_norm = (snap.rtt_ms / self.cfg.rtt_state_scale_ms).tanh();
        let mcs_norm = (snap.mcs_raw / MAX_MCS).clamp(0.0, 1.0);

        let max_idx = (self.cfg.bitrate_levels_mbps.len().saturating_sub(1)) as f32;
        let br_norm = (snap.bitrate_idx as f32 / max_idx.max(1.0)).clamp(0.0, 1.0);

        // 2. Compute Trends
        let (d_nfr, d_rtt, d_mcs) = if let Some((p_nfr, p_rtt, p_mcs)) = self.prev_norm_vals {
            (
                (nfr_norm - p_nfr).clamp(-1.0, 1.0),
                (rtt_norm - p_rtt).clamp(-1.0, 1.0),
                (mcs_norm - p_mcs).clamp(-1.0, 1.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // 3. Update history
        self.prev_norm_vals = Some((nfr_norm, rtt_norm, mcs_norm));

        // 4. One-hot encoding previous action
        let prev_decrease = if self.last_move_dir == -1 { 1.0 } else { 0.0 };
        let prev_hold = if self.last_move_dir == 0 { 1.0 } else { 0.0 };
        let prev_increase = if self.last_move_dir == 1 { 1.0 } else { 0.0 };

        // 5. Build State Tensor
        let state_vec = vec![
            nfr_norm,
            d_nfr,
            rtt_norm,
            d_rtt,
            mcs_norm,
            d_mcs,
            br_norm,
            snap.channel_busy_pct,
            snap.my_airtime_fraction,
            snap.fairness_index,
            prev_decrease,
            prev_hold,
            prev_increase,
        ];

        Tensor::from_slice(&state_vec).unsqueeze(0)
    }

    pub fn compute_reward(&mut self, snap: &EnvironmentSnapshot) -> (f32, Vec<f32>) {
        // 1. Bitrate Utility (Logarithmic Min-Max)
        let b_min = *self.cfg.bitrate_levels_mbps.first().unwrap_or(&2.0);
        let b_max = *self.cfg.bitrate_levels_mbps.last().unwrap_or(&50.0);
        let b_curr = snap.bitrate_bps / 1e6;

        let log_curr = (1.0 + b_curr).ln();
        let log_min = (1.0 + b_min).ln();
        let log_max = (1.0 + b_max).ln();

        let br_utility = if log_max > log_min {
            ((log_curr - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // 2. NFR Penalty
        let nfr_deficit = (self.cfg.nfr_target - snap.nfr).max(0.0);
        let nfr_penalty = nfr_deficit * 100.0;

        // 3. RTT Penalty
        let rtt_excess_ms = (snap.rtt_ms - self.cfg.rtt_target_ms).max(0.0);
        let rtt_penalty = rtt_excess_ms / 100.0;

        // 4. Oscillation Penalty
        let mut osc_penalty = 0.0;
        let mut current_dir = 0;

        if let Some(prev_idx) = self.prev_bitrate_idx {
            if snap.bitrate_idx > prev_idx {
                current_dir = 1;
            } else if snap.bitrate_idx < prev_idx {
                current_dir = -1;
            }

            // Check Zig-Zag: If we moved opposite to the last move
            if current_dir != 0 && current_dir == -self.last_move_dir {
                osc_penalty = 1.0;
            }
        }
        self.last_move_dir = current_dir;
        self.prev_bitrate_idx = Some(snap.bitrate_idx);

        // 5. Fairness Penalty
        let n_users = snap.active_vr_count.max(1) as f32;
        let target_share = 1.0 / n_users;
        let deviation = ((snap.my_airtime_fraction - target_share) / target_share).max(0.0);
        let fairness_penalty = deviation.powi(2);

        // 5. Calculate weighted sum
        let reward = (self.cfg.w_bitrate * br_utility)
            - (self.cfg.w_nfr * nfr_penalty)
            - (self.cfg.w_rtt * rtt_penalty)
            - (self.cfg.w_osc * osc_penalty)
            - (self.cfg.w_fairness * fairness_penalty);

        (
            reward,
            vec![
                self.cfg.w_bitrate * br_utility,
                -self.cfg.w_nfr * nfr_penalty,
                -self.cfg.w_rtt * rtt_penalty,
                -self.cfg.w_osc * osc_penalty,
                -self.cfg.w_fairness * fairness_penalty,
            ],
        )
    }
}
