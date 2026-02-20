use alvr_events::EnvironmentSnapshot;
use tch::Tensor;

pub const STATE_DIM: i64 = 15;
const MAX_MCS: f32 = 11.0; // 802.11ax/ac max index

const SWITCH_AGE_SCALE: f32 = 10.0; // 10 steps since switch -> tanh(1.0)

#[derive(Clone, Debug)]
pub struct LearningConfig {
    pub bitrate_levels_mbps: Vec<f32>, // The discrete bitrate ladder

    // Targets
    pub nfr_target: f32,
    pub rtt_target_ms: f32,

    // Action shielding (normalization limits)
    pub nfr_tolerance: f32,
    pub rtt_tolerance_ms: f32,

    // Weights
    pub w_bitrate: f32,
    pub w_nfr: f32,
    pub w_rtt: f32,
    pub w_switch: f32,
    pub w_fairness: f32,

    // Bitrate Utility
    pub use_log_bitrate: bool,

    // Max penalty clamp
    pub max_penalty_clamp: f32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            bitrate_levels_mbps: vec![2.0, 5.0, 10.0, 20.0, 50.0], // should be ordered
            nfr_target: 0.95,
            rtt_target_ms: 22.0,

            nfr_tolerance: 0.05,
            rtt_tolerance_ms: 50.0,

            w_bitrate: 1.0,
            w_nfr: 2.0,
            w_rtt: 2.0,
            w_switch: 0.05,
            w_fairness: 0.5,

            use_log_bitrate: false,

            max_penalty_clamp: 10.0,
        }
    }
}

impl LearningConfig {
    pub fn new(
        bitrate_levels_mbps: Vec<f32>,
        nfr_target: f32,
        nfr_tolerance: f32,
        rtt_target_ms: f32,
        rtt_tolerance_ms: f32,
        w_bitrate: f32,
        w_nfr: f32,
        w_rtt: f32,
        w_switch: f32,
        w_fairness: f32,
        use_log_bitrate: bool,
        max_penalty_clamp: f32,
    ) -> Self {
        Self {
            bitrate_levels_mbps,
            nfr_target,
            rtt_target_ms,
            nfr_tolerance,
            rtt_tolerance_ms,
            w_bitrate,
            w_nfr,
            w_rtt,
            w_switch,
            w_fairness,
            use_log_bitrate,
            max_penalty_clamp,
        }
    }
}

pub struct StreamingEnvironment {
    pub cfg: LearningConfig,

    pub prev_snapshot: Option<EnvironmentSnapshot>,

    pub action_momentum: i32,   // Direction of move
    pub last_switch_steps: i32, // Number of steps since last switch
}

impl StreamingEnvironment {
    pub fn new(cfg: LearningConfig) -> Self {
        Self {
            cfg,
            prev_snapshot: None,
            action_momentum: 0,
            last_switch_steps: 0,
        }
    }

    fn get_bitrate_utility(&self, snap: &EnvironmentSnapshot) -> f32 {
        let b_curr_mbps = snap.bitrate_bps / 1e6;
        let b_levels = &self.cfg.bitrate_levels_mbps;

        let b_min = *b_levels.first().unwrap();
        let b_max = *b_levels.last().unwrap();

        let safe_curr = b_curr_mbps.max(0.1);
        let safe_min = b_min.max(0.1);
        let safe_max = b_max.max(safe_min + 0.1);

        if self.cfg.use_log_bitrate {
            // Logarithmic Utility (Diminishing returns)
            let u_curr = safe_curr.ln();
            let u_min = safe_min.ln();
            let u_max = safe_max.ln();
            ((u_curr - u_min) / (u_max - u_min)).clamp(0.0, 1.0)
        } else {
            // Linear Utility
            ((safe_curr - safe_min) / (safe_max - safe_min)).clamp(0.0, 1.0)
        }
    }

    pub fn update_last_action(&mut self, a_t_idx: i64) {
        self.action_momentum = match a_t_idx {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        };

        if a_t_idx != 1 {
            self.last_switch_steps = 0;
        } else {
            self.last_switch_steps += 1;
        }
    }

    pub fn update_history(&mut self, snap: &EnvironmentSnapshot) {
        self.prev_snapshot = Some(snap.clone());
    }

    pub fn build_state_vector(&self, snap: &EnvironmentSnapshot) -> Tensor {
        // 1. Distance to Targets
        let nfr_dist = ((self.cfg.nfr_target - snap.nfr) / self.cfg.nfr_tolerance).clamp(-1.0, 1.0);
        let rtt_dist =
            ((snap.rtt_ms - self.cfg.rtt_target_ms) / self.cfg.rtt_tolerance_ms).clamp(-1.0, 1.0);

        // 2. Compute Trends
        let (d_nfr, d_rtt) = if let Some(prev_snap) = &self.prev_snapshot {
            (
                ((snap.nfr - prev_snap.nfr) / self.cfg.nfr_tolerance).clamp(-1.0, 1.0),
                ((snap.rtt_ms - prev_snap.rtt_ms) / self.cfg.rtt_tolerance_ms).clamp(-1.0, 1.0),
            )
        } else {
            (0.0, 0.0)
        };

        // 3. RTT Coefficient of Variation (CV)
        let rtt_cv = if snap.rtt_ms > 1e-5 {
            ((snap.rtt_max_ms - snap.rtt_ms) / snap.rtt_ms).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 4. Efficiency
        let target_bps = snap.bitrate_bps.max(1.0);
        let efficiency = (snap.actual_throughput_bps / target_bps).clamp(0.0, 1.0);

        // 5. Bitrate Utility
        let bitrate_utility = self.get_bitrate_utility(snap);

        // 6. History
        let time_since_switch = (self.last_switch_steps as f32 / SWITCH_AGE_SCALE).tanh();
        let action_momentum = self.action_momentum as f32;

        // 7. MCS
        let mcs = (snap.mcs_raw / MAX_MCS).clamp(0.0, 1.0);

        // Build State Tensor
        let state_vec = vec![
            // Performance
            nfr_dist,
            d_nfr,
            rtt_dist,
            d_rtt,
            rtt_cv,
            // Efficiency
            efficiency,
            // Bitrate
            bitrate_utility,
            // History
            time_since_switch,
            action_momentum,
            // Signal
            mcs,
            // Medium contention
            snap.channel_busy_pct,
            snap.tx_retry_rate,
            // Fairness
            snap.my_airtime_fraction,
            snap.fairness_index,
            // User density
            1.0 / snap.active_vr_count as f32,
        ];

        Tensor::from_slice(&state_vec).unsqueeze(0)
    }

    pub fn compute_reward(&self, snap: &EnvironmentSnapshot) -> (f32, Vec<f32>) {
        // 1. Bitrate Utility
        let br_utility = self.get_bitrate_utility(snap);

        // 2. NFR Penalty: Only punish if below target
        let nfr_deficit = (self.cfg.nfr_target - snap.nfr).max(0.0);
        let nfr_ratio = nfr_deficit / self.cfg.nfr_tolerance;
        let nfr_penalty = nfr_ratio.powi(2);

        // 3. RTT Penalty: Only punish if above target
        let rtt_excess = (snap.rtt_ms - self.cfg.rtt_target_ms).max(0.0);
        let rtt_ratio = rtt_excess / self.cfg.rtt_tolerance_ms;
        let rtt_penalty = rtt_ratio.powi(2);

        // 4. Switch Penalty
        let switch_penalty = if let Some(prev_snap) = &self.prev_snapshot {
            if prev_snap.bitrate_idx != snap.bitrate_idx {
                let prev_health_nfr = ((prev_snap.nfr
                    - (self.cfg.nfr_target - self.cfg.nfr_tolerance))
                    / self.cfg.nfr_tolerance)
                    .clamp(0.0, 1.0);
                let prev_health_rtt = (((self.cfg.rtt_target_ms + self.cfg.rtt_tolerance_ms)
                    - prev_snap.rtt_ms)
                    / self.cfg.rtt_tolerance_ms)
                    .clamp(0.0, 1.0);
                let prev_system_health = prev_health_nfr.min(prev_health_rtt);

                if snap.bitrate_idx < prev_snap.bitrate_idx {
                    prev_system_health.powi(2)
                } else {
                    1.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 5. Fairness Penalty
        let n_users = snap.active_vr_count.max(1) as f32;

        let background_traffic = (snap.channel_busy_pct - snap.total_vr_airtime_fraction).max(0.0); // % of channel activity due to non-VR traffic
        let max_safe_utilization = 0.90; // max safe channel utilization

        let available_vr_capacity = (max_safe_utilization - background_traffic).max(0.01);

        let my_fair_share = available_vr_capacity / n_users;

        let deviation = ((snap.my_airtime_fraction - my_fair_share) / my_fair_share).max(0.0);
        let fairness_penalty = deviation.powi(2);

        let clamp = |p: f32| {
            if self.cfg.max_penalty_clamp > 0.0 {
                p.min(self.cfg.max_penalty_clamp)
            } else {
                p
            }
        };

        // 6. Weighted Sum Calculation
        let reward = (self.cfg.w_bitrate * br_utility)
            - (self.cfg.w_nfr * clamp(nfr_penalty))
            - (self.cfg.w_rtt * clamp(rtt_penalty))
            - (self.cfg.w_switch * clamp(switch_penalty))
            - (self.cfg.w_fairness * clamp(fairness_penalty));

        (
            reward,
            vec![
                self.cfg.w_bitrate * br_utility,
                -self.cfg.w_nfr * clamp(nfr_penalty),
                -self.cfg.w_rtt * clamp(rtt_penalty),
                -self.cfg.w_switch * clamp(switch_penalty),
                -self.cfg.w_fairness * clamp(fairness_penalty),
            ],
        )
    }
}
