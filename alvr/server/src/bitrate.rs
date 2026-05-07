use crate::{
    ap_telemetry::{WifiMetrics, WifiStatsProcessor},
    learning_env::{LearningConfig, StreamingEnvironment, STATE_DIM},
    bravr_agent::{BravrAgent, BravrAgentConfig},
    FfiDynamicEncoderParams, FILESYSTEM_LAYOUT,
};
use alvr_common::{info, APStats, SlidingWindowAverage};
use alvr_events::{
    EnvironmentSnapshot, EventType, HeuristicStats, NominalBitrateStats, BRAVRStats,
};
use alvr_session::{
    get_profile_config, settings_schema::Switch, AveragingStrategy, BitrateAdaptiveFramerateConfig,
    BitrateConfig, BitrateMode, WindowType,
};

use std::{
    cmp::Ordering,
    collections::VecDeque,
    net::IpAddr,
    time::{Duration, Instant},
};

use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use tch::Tensor;

const UPDATE_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Default, PartialEq)]
pub struct LastNestSettings {
    max_bps: f32,
    min_bps: f32,
    bitrate_step_count: usize,
}

pub struct BitrateManager {
    client_ip: IpAddr,

    nominal_frame_interval: Duration,
    frame_interval_average: SlidingWindowAverage<Duration>,
    // note: why packet_sizes_bits_history is a queue and not a sliding average? Because some
    // network samples will be dropped but not any packet size sample
    packet_sizes_bits_history: VecDeque<(Duration, usize)>,
    encoder_latency_average: SlidingWindowAverage<Duration>,
    network_latency_average: SlidingWindowAverage<Duration>,
    bitrate_average: SlidingWindowAverage<f32>,
    decoder_latency_overstep_count: usize,
    last_frame_instant: Instant,
    last_update_instant: Instant,
    dynamic_max_bitrate: f32,
    previous_config: Option<BitrateConfig>,
    update_needed: bool,

    prev_target_bitrate_bps: f32,
    last_target_bitrate_bps: f32,
    update_interval_s: Duration,

    // Global averages (for logging/gui/nestvr)
    rtt_average: SlidingWindowAverage<Duration>,
    peak_throughput_average: SlidingWindowAverage<f32>,
    frame_interarrival_s_average: SlidingWindowAverage<f32>,

    bitrate_ladder_bps: Option<Vec<f32>>,
    bitrate_step_size_bps: f32,
    last_nest_settings: LastNestSettings,

    // Learning interval accumulators (for bravr)
    rtt_samples_s: Vec<f32>,                // rtt
    frame_interval_samples_s: Vec<f32>,     // tx frame interval
    frame_interarrival_samples_s: Vec<f32>, // rx frame interval
    total_rx_bits: f32,                     // total rx bits
    total_frame_interarrival_s: f32,        // total rx frame interval

    // Reinforcement learning components
    bravr_agent: Option<BravrAgent>,
    env: Option<StreamingEnvironment>,
    bravr_learning_enabled: bool,

    // AP & WiFi handling
    max_ap_history: usize,
    ap_stats_buffer: VecDeque<APStats>,
    wifi_processor: WifiStatsProcessor,
}

impl BitrateManager {
    pub fn new(
        max_history_size: Option<usize>,
        initial_framerate: f32,
        initial_bitrate: f32,
        history_interval: Option<Duration>,
        ewma_weight_val: Option<f32>,
        client_ip: IpAddr,
    ) -> Self {
        let max_ap_history = max_history_size.unwrap_or(10);

        Self {
            client_ip: client_ip,

            nominal_frame_interval: Duration::from_secs_f32(1. / initial_framerate),
            frame_interval_average: SlidingWindowAverage::new(
                Duration::from_millis(16),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            packet_sizes_bits_history: VecDeque::new(),
            encoder_latency_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            network_latency_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            bitrate_average: SlidingWindowAverage::new(
                initial_bitrate * 1e6,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            decoder_latency_overstep_count: 0,
            last_frame_instant: Instant::now(),
            last_update_instant: Instant::now(),
            dynamic_max_bitrate: f32::MAX,
            previous_config: None,
            update_needed: true,

            prev_target_bitrate_bps: initial_bitrate * 1e6,
            last_target_bitrate_bps: initial_bitrate * 1e6,
            update_interval_s: UPDATE_INTERVAL,

            rtt_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            peak_throughput_average: SlidingWindowAverage::new(
                300E6,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            frame_interarrival_s_average: SlidingWindowAverage::new(
                1. / initial_framerate,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),

            rtt_samples_s: Vec::with_capacity(200), // 200 samples is much more than enough
            frame_interval_samples_s: Vec::with_capacity(200),
            frame_interarrival_samples_s: Vec::with_capacity(200),
            total_rx_bits: 0.0,
            total_frame_interarrival_s: 0.0,

            bitrate_ladder_bps: None,
            bitrate_step_size_bps: 0.0,
            last_nest_settings: LastNestSettings::default(),

            bravr_agent: None,
            env: None,
            bravr_learning_enabled: false,

            max_ap_history,
            ap_stats_buffer: VecDeque::with_capacity(max_ap_history),

            wifi_processor: WifiStatsProcessor::new(client_ip),
        }
    }

    pub fn update_nominal_frame_interval(&mut self, fps: f32) {
        self.nominal_frame_interval = Duration::from_secs_f32(1. / fps);
    }

    // Note: This is used to calculate the framerate/frame interval. The frame present is the most
    // accurate event for this use.
    pub fn report_frame_present(&mut self, config: &Switch<BitrateAdaptiveFramerateConfig>) {
        let now = Instant::now();

        let interval = now - self.last_frame_instant;
        self.last_frame_instant = now;

        self.frame_interval_average.submit_sample(interval);

        self.frame_interval_samples_s.push(interval.as_secs_f32());

        if let Some(config) = config.as_option() {
            let interval_ratio =
                interval.as_secs_f32() / self.frame_interval_average.get_average().as_secs_f32();

            if interval_ratio > config.framerate_reset_threshold_multiplier
                || interval_ratio < 1.0 / config.framerate_reset_threshold_multiplier
            {
                // Clear most of the samples, keep some for stability
                self.frame_interval_average.retain(5);
                self.update_needed = true;
            }
        }
    }

    pub fn report_frame_encoded(
        &mut self,
        timestamp: Duration,
        encoder_latency: Duration,
        size_bytes: usize,
    ) {
        self.encoder_latency_average.submit_sample(encoder_latency);

        self.packet_sizes_bits_history
            .push_back((timestamp, size_bytes * 8));
    }

    pub fn report_network_statistics(
        &mut self,
        network_rtt: Duration,
        peak_throughput_bps: f32,
        frame_interarrival_s: f32,
        rx_bytes: u32,
    ) {
        self.rtt_average.submit_sample(network_rtt);

        self.peak_throughput_average
            .submit_sample(peak_throughput_bps);

        self.frame_interarrival_s_average
            .submit_sample(frame_interarrival_s);

        self.rtt_samples_s.push(network_rtt.as_secs_f32());

        self.frame_interarrival_samples_s.push(frame_interarrival_s);

        self.total_rx_bits += rx_bytes as f32 * 8.0;
        self.total_frame_interarrival_s += frame_interarrival_s;
    }

    pub fn report_ap_statistics(&mut self, ap_stats: &APStats) {
        if self.ap_stats_buffer.len() >= self.max_ap_history {
            self.ap_stats_buffer.pop_front();
        }
        self.ap_stats_buffer.push_back(ap_stats.clone());
    }

    pub fn report_frame_latencies(
        &mut self,
        config: &BitrateMode,
        timestamp: Duration,
        network_latency: Duration,
        decoder_latency: Duration,
    ) {
        if network_latency.is_zero() {
            return;
        }
        self.network_latency_average.submit_sample(network_latency);

        while let Some(&(timestamp_, size_bits)) = self.packet_sizes_bits_history.front() {
            if timestamp_ == timestamp {
                self.bitrate_average
                    .submit_sample(size_bits as f32 / network_latency.as_secs_f32());

                self.packet_sizes_bits_history.pop_front();

                break;
            } else {
                self.packet_sizes_bits_history.pop_front();
            }
        }

        if let BitrateMode::Adaptive {
            decoder_latency_limiter: Switch::Enabled(config),
            ..
        } = &config
        {
            if decoder_latency > Duration::from_millis(config.max_decoder_latency_ms) {
                self.decoder_latency_overstep_count += 1;

                if self.decoder_latency_overstep_count == config.latency_overstep_frames {
                    self.dynamic_max_bitrate =
                        f32::min(self.bitrate_average.get_average(), self.dynamic_max_bitrate)
                            * config.latency_overstep_multiplier;

                    self.update_needed = true;

                    self.decoder_latency_overstep_count = 0;
                }
            } else {
                self.decoder_latency_overstep_count = 0;
            }
        }
    }

    fn get_interval_rtt_ms_stats(&self) -> (f32, f32) {
        // Returns (Mean, Max)
        let mut sum = 0.0f32;
        let mut max = 0.0f32;
        let mut count = 0u32;

        for &rtt in &self.rtt_samples_s {
            if !rtt.is_finite() {
                continue;
            }

            sum += rtt;
            max = max.max(rtt);
            count += 1;
        }

        if count == 0 {
            return (0.0, 0.0);
        }

        let mean = sum / count as f32;
        (mean * 1000.0, max * 1000.0)
    }

    fn get_interval_nfr(&self) -> f32 {
        if self.frame_interval_samples_s.is_empty() || self.frame_interarrival_samples_s.is_empty()
        {
            return 0.0;
        }

        let count_tx = self.frame_interval_samples_s.len() as f32;
        let sum_tx: f32 = self.frame_interval_samples_s.iter().sum();
        let fps_tx = count_tx / sum_tx;

        let count_rx = self.frame_interarrival_samples_s.len() as f32;
        let sum_rx: f32 = self.frame_interarrival_samples_s.iter().sum();
        let fps_rx = count_rx / sum_rx;

        // Safety check: if there is little transmitted data, do not judge the network
        if sum_tx < 0.2 * self.update_interval_s.as_secs_f32() {
            return 1.0;
        }
        // Safety check: if there is little reception data, assume bad network and return 0
        if sum_rx < 0.2 * self.update_interval_s.as_secs_f32() {
            return 0.0;
        }

        if fps_tx > 0.0 {
            (fps_rx / fps_tx).clamp(0.0, 3.0)
        } else {
            0.0
        }
    }

    fn get_interval_throughput_bps(&self) -> f32 {
        if self.total_frame_interarrival_s == 0.0 {
            return 0.0;
        }
        self.total_rx_bits / self.total_frame_interarrival_s
    }

    fn get_action_mask(
        env: &StreamingEnvironment,
        snap: &EnvironmentSnapshot,
        current_idx: usize,
        ladder_len: usize,
        shielding: bool,
    ) -> Vec<bool> {
        // 0 = Decrease, 1 = Hold, 2 = Increase
        let mut mask = vec![true, true, true];

        // Rule 1: Physical boundaries (cannot go below minimum or above maximum)
        if current_idx == 0 {
            mask[0] = false;
        } else if current_idx == ladder_len - 1 {
            mask[2] = false;
        }

        // RULE 2: Safety Shielding
        if shielding {
            let is_emergency = snap.rtt_ms > env.cfg.rtt_target_ms + env.cfg.rtt_tolerance_ms
                || snap.nfr < env.cfg.nfr_target - env.cfg.nfr_tolerance;

            if is_emergency {
                // Strictly forbid increase
                mask[2] = false;
            }
        }

        mask
    }

    fn reset_sample_stats(&mut self) {
        self.rtt_samples_s.clear();
        self.frame_interval_samples_s.clear();
        self.frame_interarrival_samples_s.clear();

        self.total_rx_bits = 0.0;
        self.total_frame_interarrival_s = 0.0;
    }

    pub fn get_encoder_params(
        &mut self,
        config: &BitrateConfig,
    ) -> (FfiDynamicEncoderParams, Option<NominalBitrateStats>) {
        let now = Instant::now();

        if self
            .previous_config
            .as_ref()
            .map(|prev| config != prev)
            .unwrap_or(true)
        {
            self.previous_config = Some(config.clone());

            let mut max_history_size = Some(256);
            let mut history_interval = None;
            let mut ewma_weight_val = None;

            match &config.mode {
                BitrateMode::NestVr {
                    max_bitrate_mbps,
                    min_bitrate_mbps,
                    initial_bitrate_mbps,
                    averaging_strategy,
                    nest_vr_profile,
                    ..
                } => {
                    let profile_config = get_profile_config(
                        *max_bitrate_mbps,
                        *min_bitrate_mbps,
                        *initial_bitrate_mbps,
                        nest_vr_profile,
                    );

                    self.update_interval_s =
                        Duration::from_secs_f32(profile_config.update_interval_nestvr_s);

                    match averaging_strategy {
                        AveragingStrategy::SimpleWindowAverage { window_type, .. } => {
                            match window_type {
                                WindowType::BySeconds {
                                    sliding_window_secs,
                                    ..
                                } => {
                                    history_interval = Some(Duration::from_secs_f32(
                                        sliding_window_secs
                                            .unwrap_or(profile_config.update_interval_nestvr_s),
                                    ));

                                    max_history_size = None;
                                }
                                WindowType::BySamples {
                                    sliding_window_samp,
                                    ..
                                } => {
                                    max_history_size = Some(*sliding_window_samp);
                                }
                            }
                        }
                        AveragingStrategy::ExponentialMovingAverage { ewma_weight, .. } => {
                            ewma_weight_val = Some(*ewma_weight);
                        }
                    }
                }
                BitrateMode::Adaptive { history_size, .. } => {
                    self.update_interval_s = UPDATE_INTERVAL;

                    max_history_size = Some(*history_size);
                }
                BitrateMode::Bravr {
                    update_interval_s,
                    bitrate_levels_mbps,
                    nfr_target,
                    nfr_tolerance,
                    rtt_target_ms,
                    rtt_tolerance_ms,
                    w_bitrate,
                    w_nfr,
                    w_rtt,
                    w_switch,
                    w_fairness,
                    use_log_bitrate,
                    max_penalty_clamp,
                    agent_config,
                    ..
                } => {
                    self.update_interval_s = Duration::from_secs_f32(*update_interval_s);

                    let mut bitrate_levels_mbps = bitrate_levels_mbps.clone();
                    bitrate_levels_mbps.sort_by(|a, b| a.partial_cmp(b).unwrap()); // sort asc

                    let env_config = LearningConfig::new(
                        bitrate_levels_mbps.clone(),
                        *nfr_target,
                        *nfr_tolerance,
                        *rtt_target_ms,
                        *rtt_tolerance_ms,
                        *w_bitrate,
                        *w_nfr,
                        *w_rtt,
                        *w_switch,
                        *w_fairness,
                        *use_log_bitrate,
                        *max_penalty_clamp,
                    );
                    self.env = Some(StreamingEnvironment::new(env_config));

                    let model_path_buf = FILESYSTEM_LAYOUT.bravr_model();
                    let state_dim = STATE_DIM;
                    self.bravr_agent = Some(BravrAgent::new(BravrAgentConfig {
                        gamma: agent_config.gamma,
                        lr: agent_config.lr,
                        tau: agent_config.tau,
                        temperature: agent_config.temperature,
                        n_step: agent_config.n_step,
                        epsilon: agent_config.epsilon,
                        state_dim,
                        hidden_dim: agent_config.hidden_dim as i64,
                        action_shielding_enabled: agent_config.action_shielding_enabled,
                        ap_info_enabled: agent_config.ap_info_enabled,
                        model_path: model_path_buf,
                        load_model: agent_config.load_model,
                        save_model: agent_config.save_model,
                    }));

                    self.enable_bravr_learning();
                }
                _ => {
                    self.update_interval_s = UPDATE_INTERVAL;
                }
            }
            let averages_dur = [
                &mut self.frame_interval_average,
                &mut self.encoder_latency_average,
                &mut self.network_latency_average,
                &mut self.rtt_average,
            ];
            let averages_f32 = [
                &mut self.bitrate_average,
                &mut self.peak_throughput_average,
                &mut self.frame_interarrival_s_average,
            ];

            for average in averages_dur {
                average.update_max_history_size(max_history_size);
                average.update_history_interval(history_interval);
                average.update_ewma_weight(ewma_weight_val);
            }
            for average in averages_f32 {
                average.update_max_history_size(max_history_size);
                average.update_history_interval(history_interval);
                average.update_ewma_weight(ewma_weight_val);
            }
        } else if !self.update_needed
            && (now < (self.last_update_instant + self.update_interval_s)
                || matches!(config.mode, BitrateMode::ConstantMbps(_)))
        {
            return (
                FfiDynamicEncoderParams {
                    updated: 0,
                    bitrate_bps: 0,
                    framerate: 0.0,
                },
                None,
            );
        }

        self.last_update_instant = now;
        self.update_needed = false;

        // Interval Averages (raw data between decisions)
        let (rtt_ms, rtt_max_ms) = self.get_interval_rtt_ms_stats();
        let nfr = self.get_interval_nfr();
        let actual_throughput_bps = self.get_interval_throughput_bps();
        let wifi_metrics_raw = self.wifi_processor.process(&self.ap_stats_buffer);

        // Build Snapshot
        let raw_snap = EnvironmentSnapshot {
            bitrate_idx: None,
            bitrate_bps: self.last_target_bitrate_bps,
            nfr,
            rtt_ms,
            rtt_max_ms,
            actual_throughput_bps,
            mcs_raw: wifi_metrics_raw.mcs_raw,
            channel_busy_pct: wifi_metrics_raw.channel_busy_pct,
            tx_retry_rate: wifi_metrics_raw.tx_retry_rate,
            my_airtime_fraction: wifi_metrics_raw.my_airtime_fraction,
            my_airtime_share: wifi_metrics_raw.my_airtime_share,
            total_vr_airtime_fraction: wifi_metrics_raw.total_vr_airtime_fraction,
            active_vr_count: wifi_metrics_raw.active_vr_count,
            fairness_index: wifi_metrics_raw.fairness_index,
        };

        // Environment snapshot is always logged for debugging
        alvr_events::send_event(EventType::EnvironmentSnapshot(raw_snap.clone()));

        let mut stats = NominalBitrateStats::default();

        let bitrate_bps = match &config.mode {
            BitrateMode::ConstantMbps(bitrate_mbps) => *bitrate_mbps as f32 * 1e6,
            BitrateMode::NestVr {
                max_bitrate_mbps,
                min_bitrate_mbps,
                initial_bitrate_mbps,
                nest_vr_profile,
                ..
            } => {
                pub fn upper_bound_bitrate(bitrate_bps: f32, bitrate_ladder: &Vec<f32>) -> f32 {
                    // Perform binary search to find the largest value less than or equal to `bitrate_bps`
                    match bitrate_ladder
                        .binary_search_by(|x| x.partial_cmp(&bitrate_bps).unwrap_or(Ordering::Less))
                    {
                        Ok(index) => bitrate_ladder[index], // Exact match found
                        Err(index) => {
                            // If not found, `index` is where the value would be inserted to maintain sorted order
                            if index == 0 {
                                // If `bitrate_bps` is smaller than the first element, return the first element
                                bitrate_ladder.first().copied().unwrap_or(bitrate_bps)
                            } else {
                                // Otherwise, return the element just before the insertion point (i.e., the largest <= bitrate_bps)
                                bitrate_ladder[index - 1]
                            }
                        }
                    }
                }
                fn minmax_bitrate(
                    bitrate_bps: f32,
                    max_bitrate_bps: f32,
                    min_bitrate_bps: f32,
                ) -> f32 {
                    let mut bitrate = bitrate_bps;

                    bitrate = f32::min(bitrate, max_bitrate_bps);
                    bitrate = f32::max(bitrate, min_bitrate_bps);

                    bitrate
                }

                let profile_config = get_profile_config(
                    *max_bitrate_mbps,
                    *min_bitrate_mbps,
                    *initial_bitrate_mbps,
                    nest_vr_profile,
                );

                let mut recompute_bitrate_ladder = false;

                let current_settings = LastNestSettings {
                    max_bps: max_bitrate_mbps * 1E6,
                    min_bps: min_bitrate_mbps * 1E6,
                    bitrate_step_count: profile_config.bitrate_step_count,
                };

                if current_settings != self.last_nest_settings {
                    recompute_bitrate_ladder = true;
                }

                // ensure max is bigger than min
                let (min_bps, max_bps) = if min_bitrate_mbps > max_bitrate_mbps {
                    (max_bitrate_mbps * 1E6, min_bitrate_mbps * 1E6)
                } else {
                    (min_bitrate_mbps * 1E6, max_bitrate_mbps * 1E6)
                };

                if self.bitrate_ladder_bps.is_none() || recompute_bitrate_ladder == true {
                    let bitrate_step_count = profile_config.bitrate_step_count;

                    if max_bps != 0.0 && min_bps != 0.0 {
                        let mut vec_bitrates = Vec::new();

                        let bitrate_step_size_bps = (max_bps - min_bps) / bitrate_step_count as f32;

                        let mut last_value = min_bps;

                        vec_bitrates.push(min_bps); // first bitrate is min
                        for _ in 0..bitrate_step_count {
                            last_value += bitrate_step_size_bps;
                            vec_bitrates.push(last_value);
                        }

                        self.bitrate_ladder_bps = Some(vec_bitrates);
                        self.bitrate_step_size_bps = bitrate_step_size_bps;

                        self.last_target_bitrate_bps = upper_bound_bitrate(
                            self.last_target_bitrate_bps,
                            &self.bitrate_ladder_bps.clone().unwrap(),
                        );
                    }
                }

                // Sample from uniform distribution
                let mut rng = thread_rng();
                let uniform_dist = Uniform::new(0.0, 1.0);

                let r_rtt = rng.sample(uniform_dist);
                let r_inc = rng.sample(uniform_dist);

                let frame_interval_s = self.frame_interval_average.get_average().as_secs_f32();

                let fps_tx_avg = if frame_interval_s != 0.0 {
                    1.0 / frame_interval_s
                } else {
                    0.0
                };
                let fps_rx_avg = if self.frame_interarrival_s_average.get_average() != 0.0 {
                    1.0 / self.frame_interarrival_s_average.get_average()
                } else {
                    0.0
                };

                let nfr_avg = fps_rx_avg / fps_tx_avg;
                let rtt_avg_ms = self.rtt_average.get_average().as_secs_f32() * 1000.0;

                let estimated_capacity_bps = self.peak_throughput_average.get_average();

                let mut bitrate_bps: f32 = self.last_target_bitrate_bps;

                if nfr_avg < profile_config.nfr_thresh {
                    // decrease
                    bitrate_bps -=
                        profile_config.bitrate_dec_steps as f32 * self.bitrate_step_size_bps;
                } else {
                    if rtt_avg_ms > profile_config.rtt_thresh_ms {
                        if r_rtt <= profile_config.rtt_adj_prob {
                            // decrease
                            bitrate_bps -= profile_config.bitrate_dec_steps as f32
                                * self.bitrate_step_size_bps;
                        }
                    } else {
                        if r_inc <= profile_config.bitrate_inc_prob {
                            // increase
                            bitrate_bps += profile_config.bitrate_inc_steps as f32
                                * self.bitrate_step_size_bps;
                        }
                    }
                }

                // Ensure bitrate is below the estimated network capacity
                let capacity_upper_limit =
                    profile_config.capacity_scaling_factor * estimated_capacity_bps;

                bitrate_bps = f32::min(bitrate_bps, capacity_upper_limit);

                // Ensure bitrate is always within the configured range
                bitrate_bps = minmax_bitrate(bitrate_bps, max_bps, min_bps);

                bitrate_bps =
                    upper_bound_bitrate(bitrate_bps, &self.bitrate_ladder_bps.clone().unwrap());

                let heur_stats = HeuristicStats {
                    bitrate_step_count: profile_config.bitrate_step_count,

                    bitrate_dec_steps: profile_config.bitrate_dec_steps,
                    bitrate_inc_steps: profile_config.bitrate_inc_steps,

                    bitrate_step_size_bps: self.bitrate_step_size_bps,

                    r_rtt: r_rtt,
                    r_inc: r_inc,

                    rtt_adj_prob: profile_config.rtt_adj_prob,
                    bitrate_inc_prob: profile_config.bitrate_inc_prob,

                    fps_tx_avg: fps_tx_avg,
                    fps_rx_avg: fps_rx_avg,

                    nfr_avg: nfr_avg,
                    rtt_avg_ms: rtt_avg_ms,

                    nfr_thresh: profile_config.nfr_thresh,
                    rtt_thresh_ms: profile_config.rtt_thresh_ms,

                    requested_bitrate_bps: bitrate_bps,
                };
                alvr_events::send_event(EventType::HeuristicStats(heur_stats));

                stats.manual_max_bps = Some(max_bitrate_mbps * 1E6);
                stats.manual_min_bps = Some(min_bitrate_mbps * 1E6);

                bitrate_bps
            }

            BitrateMode::Adaptive {
                saturation_multiplier,
                max_bitrate_mbps,
                min_bitrate_mbps,
                max_network_latency_ms,
                encoder_latency_limiter,
                ..
            } => {
                let initial_bitrate_average_bps = self.bitrate_average.get_average();

                let mut bitrate_bps = initial_bitrate_average_bps * saturation_multiplier;
                stats.scaled_calculated_bps = Some(bitrate_bps);

                bitrate_bps = f32::min(bitrate_bps, self.dynamic_max_bitrate);
                stats.decoder_latency_limiter_bps = Some(self.dynamic_max_bitrate);

                if let Switch::Enabled(max_ms) = max_network_latency_ms {
                    let max = initial_bitrate_average_bps * (*max_ms as f32 / 1000.0)
                        / self.network_latency_average.get_average().as_secs_f32();
                    bitrate_bps = f32::min(bitrate_bps, max);

                    stats.network_latency_limiter_bps = Some(max);
                }

                if let Switch::Enabled(config) = encoder_latency_limiter {
                    let saturation = self.encoder_latency_average.get_average().as_secs_f32()
                        / self.nominal_frame_interval.as_secs_f32();
                    let max =
                        initial_bitrate_average_bps * config.max_saturation_multiplier / saturation;
                    stats.encoder_latency_limiter_bps = Some(max);

                    if saturation > config.max_saturation_multiplier {
                        // Note: this assumes linear relationship between bitrate and encoder
                        // latency but this may not be the case
                        bitrate_bps = f32::min(bitrate_bps, max);
                    }
                }

                if let Switch::Enabled(max) = max_bitrate_mbps {
                    let max = *max as f32 * 1e6;
                    bitrate_bps = f32::min(bitrate_bps, max);

                    stats.manual_max_bps = Some(max);
                }
                if let Switch::Enabled(min) = min_bitrate_mbps {
                    let min = *min as f32 * 1e6;
                    bitrate_bps = f32::max(bitrate_bps, min);

                    stats.manual_min_bps = Some(min);
                }

                bitrate_bps
            }
            BitrateMode::Bravr { .. } => {
                if self.bravr_learning_enabled {
                    // Warmup period: if there is no valid data, return the current bitrate and wait for the next interval
                    if rtt_ms < 1.0 {
                        self.last_target_bitrate_bps
                    } else {
                        if let (Some(agent), Some(env)) = (&mut self.bravr_agent, &mut self.env) {
                            // Agent snapshot enforces partial observability (depending on whether AP info use is enabled)
                            let mut agent_snap =
                                raw_snap.masked_for_agent(agent.cfg.ap_info_enabled);

                            let current_bitrate_mbps = self.last_target_bitrate_bps / 1e6;
                            let current_bitrate_idx = env
                                .cfg
                                .bitrate_levels_mbps
                                .iter()
                                .position(|&b| (b as f32 - current_bitrate_mbps as f32).abs() < 0.1)
                                .unwrap_or(0);

                            agent_snap.bitrate_idx = Some(current_bitrate_idx);

                            // RL Step
                            // Calculate reward and update history
                            let (reward, r_components) = env.compute_reward(&agent_snap); // r_t = r(s_{t-1}, a_{t-1})

                            let s_t = env.build_state_vector(&agent_snap); // Build current state

                            env.update_history(&agent_snap);

                            // Log previous state and action since update() overrides them
                            let vec_to_str = |t: &Option<Tensor>| -> String {
                                match t {
                                    Some(tensor) => {
                                        Vec::<f32>::try_from(tensor.view([-1]).shallow_clone())
                                            .map(|v| format!("{:?}", v))
                                            .unwrap_or_else(|_| "[]".to_string())
                                    }
                                    None => "[]".to_string(),
                                }
                            };
                            let s_prev_str = vec_to_str(&agent.s_prev);
                            let a_prev_idx = agent.a_prev_idx;

                            let ladder_len = env.cfg.bitrate_levels_mbps.len();
                            let action_mask = Self::get_action_mask(
                                &env,
                                &agent_snap,
                                current_bitrate_idx,
                                ladder_len,
                                agent.cfg.action_shielding_enabled,
                            );

                            // Select action
                            let (a_t_idx, q_values, action_probs, policy_entropy, matches_argmax) =
                                agent.select_action(&s_t, &action_mask);
                            // Perform SARSA update (s_{t-1}, a_{t-1}, r_{t-1}, s_t, a_t)
                            let td_error = agent.update(&s_t, a_t_idx, reward, &action_mask);

                            env.update_last_action(a_t_idx);

                            // Apply action
                            let next_bitrate_idx = match a_t_idx {
                                0 => current_bitrate_idx.saturating_sub(1),
                                1 => current_bitrate_idx,
                                2 => current_bitrate_idx + 1,
                                _ => current_bitrate_idx,
                            };
                            let new_bitrate_bps = env.cfg.bitrate_levels_mbps
                                [next_bitrate_idx.clamp(0, ladder_len - 1)]
                                * 1e6;

                            // 6. Stats
                            let bravr_stats = BRAVRStats {
                                s_prev: s_prev_str,                          // previous state
                                a_prev_idx: a_prev_idx, // previous action index
                                r_prev: reward,         // previous reward
                                s_t: vec_to_str(&Some(s_t.shallow_clone())), // current state
                                a_t_idx,                // current action index
                                matches_argmax,         // whether current action matches argmax
                                r_components,           // reward components
                                q_values,               // q values
                                action_probs,           // action probabilities
                                policy_entropy,         // policy entropy
                                td_error,               // td error
                                requested_bitrate_bps: new_bitrate_bps, // requested bitrate
                            };

                            alvr_events::send_event(EventType::BRAVRStats(bravr_stats));

                            new_bitrate_bps
                        } else {
                            self.last_target_bitrate_bps
                        }
                    }
                } else {
                    self.last_target_bitrate_bps
                }
            }
        };

        stats.requested_bps = bitrate_bps;
        self.prev_target_bitrate_bps = self.last_target_bitrate_bps;
        self.last_target_bitrate_bps = bitrate_bps;

        let frame_interval = if config.adapt_to_framerate.enabled() {
            self.frame_interval_average.get_average()
        } else {
            self.nominal_frame_interval
        };

        // Clear buffer and reset interval accumulators
        self.ap_stats_buffer.clear();
        self.reset_sample_stats();

        (
            FfiDynamicEncoderParams {
                updated: 1,
                bitrate_bps: bitrate_bps as u64,
                framerate: 1.0 / frame_interval.as_secs_f32().min(1.0),
            },
            Some(stats),
        )
    }

    pub fn save_bravr_model(&mut self) {
        if let Some(agent) = self.bravr_agent.as_mut() {
            agent.finish_episode();
            info!("BRAVR: Saving model on disconnect (if enabled)...");
            agent.save_to_disk();
        }
    }

    pub fn disable_bravr_learning(&mut self) {
        self.bravr_learning_enabled = false;
        info!("BRAVR learning disabled for client {}", self.client_ip);
    }

    pub fn enable_bravr_learning(&mut self) {
        self.bravr_learning_enabled = true;
        info!("BRAVR learning (re-)enabled for client {}", self.client_ip);
    }
}

impl Drop for BitrateManager {
    fn drop(&mut self) {
        if let Some(agent) = self.bravr_agent.as_mut() {
            agent.finish_episode();
            info!("BRAVR: BitrateManager dropping. Saving model (if enabled)...");
            agent.save_to_disk();
        }
    }
}
