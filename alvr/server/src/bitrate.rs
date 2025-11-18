use crate::{sarsa_agent::SarsaAgent, sarsa_agent::SarsaAgentConfig, FfiDynamicEncoderParams};
use alvr_common::{warn, APStats, Client, Interface, SlidingWindowAverage};
use alvr_events::{EventType, HeuristicStats, NominalBitrateStats, SARSAStats};
use alvr_session::{
    get_profile_config, settings_schema::Switch, AveragingStrategy, BitrateAdaptiveFramerateConfig,
    BitrateConfig, BitrateMode, WindowType,
};
use std::{
    collections::VecDeque,
    net::IpAddr,
    time::{Duration, Instant},
};

use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use tch::Tensor;

const UPDATE_INTERVAL: Duration = Duration::from_secs(1);

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

    rtt_average: SlidingWindowAverage<Duration>,
    peak_throughput_average: SlidingWindowAverage<f32>,
    frame_interarrival_average: SlidingWindowAverage<f32>,

    sarsa_agent: Option<SarsaAgent>,
    max_ap_history: usize,
    ap_stats_buffer: VecDeque<APStats>,
    prev_busy_time_ms: Option<f32>,
    prev_active_time_ms: Option<f32>,
    prev_state_vector: Option<Vec<f32>>,
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
            frame_interarrival_average: SlidingWindowAverage::new(
                1. / initial_framerate,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),

            sarsa_agent: None,
            max_ap_history,
            ap_stats_buffer: VecDeque::with_capacity(max_ap_history),
            prev_busy_time_ms: Some(0.0),
            prev_active_time_ms: Some(0.0),
            prev_state_vector: None,
        }
    }

    pub fn find_client_interface(
        &self,
        ap_stats: &APStats,
        client_ip: IpAddr,
    ) -> (Option<Interface>, Option<Client>) {
        let mut interface = None;
        let mut client_ap_stats = None;

        for iface in &ap_stats.interfaces {
            for client in &iface.clients {
                if client.ip.parse::<IpAddr>().ok() == Some(client_ip) {
                    interface = Some(iface.clone());
                    client_ap_stats = Some(client.clone());
                    break;
                }
            }
            if client_ap_stats.is_some() {
                break;
            }
        }
        (interface, client_ap_stats)
    }

    pub fn build_state_vector(&mut self) -> Tensor {
        // Builds the current state vector for the SARSA agent, combining streaming and AP-level statistics.
        // Features are normalized to [0,1] for stable NN training.
        let sarsa_cfg = &self.sarsa_agent.as_ref().unwrap().cfg;

        // 1. Normalized average NFR (frame delivery ratio)
        let frame_interval_s = self.frame_interval_average.get_average().as_secs_f32();
        let fps_tx_avg = if frame_interval_s > 0.0 {
            1.0 / frame_interval_s
        } else {
            0.0
        };
        let fps_rx_avg = if self.frame_interarrival_average.get_average() > 0.0 {
            1.0 / self.frame_interarrival_average.get_average()
        } else {
            0.0
        };
        let nfr_avg = if fps_tx_avg > 0.0 {
            (fps_rx_avg / fps_tx_avg).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // 2. Normalized average VF-RTT
        let rtt_avg = self.rtt_average.get_average().as_secs_f32().min(1.0);

        // 3. Normalized bitrate utility
        let bitrate_util =
            (self.last_target_bitrate_bps / (sarsa_cfg.max_bitrate_mbps * 1e6)).clamp(0.0, 1.0);

        // 5. AP statistics
        let (mcs_avg, airtime_avg) = if !self.ap_stats_buffer.is_empty() {
            let mut mcs_sum = 0.0;
            let mut mcs_count = 0.0;
            let mut airtime_utilization = 0.0;

            let last_idx = self.ap_stats_buffer.len().saturating_sub(1);
            for (i, ap_stats) in self.ap_stats_buffer.iter().enumerate() {
                let (iface_opt, client_opt) = self.find_client_interface(ap_stats, self.client_ip);

                if let Some(client) = client_opt {
                    if let Ok(mcs) = client.rx.mcs.parse::<f32>() {
                        mcs_sum += mcs;
                        mcs_count += 1.0;
                    }
                }

                if let Some(iface) = iface_opt {
                    // Airtime utilization
                    if i == last_idx {
                        let busy_delta = iface.ch_busy_time_ms.parse::<f32>().unwrap_or(0.0)
                            - self.prev_busy_time_ms.unwrap_or(0.0);
                        let active_delta = iface.ch_active_time_ms.parse::<f32>().unwrap_or(1.0)
                            - self.prev_active_time_ms.unwrap_or(0.0);

                        airtime_utilization = if active_delta > 0.0 {
                            (busy_delta / active_delta).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };

                        self.prev_busy_time_ms = iface.ch_busy_time_ms.parse::<f32>().ok();
                        self.prev_active_time_ms = iface.ch_active_time_ms.parse::<f32>().ok();
                    }
                }
            }

            self.ap_stats_buffer.clear();

            let mcs_avg_norm = if mcs_count > 0.0 {
                (mcs_sum / mcs_count) / 11.0
            } else {
                0.0
            };

            (mcs_avg_norm, airtime_utilization)
        } else if let Some(prev) = &self.prev_state_vector {
            let len = prev.len();
            if len >= 5 {
                (prev[len - 2], prev[len - 1])
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        // 6. Build the final state vector
        let state_vec = vec![nfr_avg, rtt_avg, bitrate_util, mcs_avg, airtime_avg];

        assert_eq!(
            state_vec.len(),
            self.sarsa_agent.as_ref().unwrap().cfg.state_dim as usize,
            "SARSA state_dim mismatch!"
        );

        self.prev_state_vector = Some(state_vec.clone());

        let state_tensor = Tensor::f_from_slice(&state_vec)
            .expect("Failed to create tensor from slice")
            .unsqueeze(0);

        state_tensor // shape [1, state_dim]
    }

    pub fn compute_reward(&mut self) -> f32 {
        // Extract SARSA config
        let sarsa_cfg = &self.sarsa_agent.as_ref().unwrap().cfg;

        // 1. Bitrate utility (logarithmic)
        let b_min_bps = sarsa_cfg.min_bitrate_mbps * 1e6;
        let b_max_bps = sarsa_cfg.max_bitrate_mbps * 1e6;

        let bitrate_norm = ((self.last_target_bitrate_bps as f32) - b_min_bps as f32)
            / ((b_max_bps - b_min_bps) as f32);
        let bitrate_norm = bitrate_norm.clamp(0.0, 1.0);

        // 2. NFR
        let frame_interval_s = self.frame_interval_average.get_average().as_secs_f32();
        let fps_tx_avg = if frame_interval_s > 0.0 {
            1.0 / frame_interval_s
        } else {
            0.0
        };
        let fps_rx_avg = if self.frame_interarrival_average.get_average() != 0.0 {
            1.0 / self.frame_interarrival_average.get_average()
        } else {
            0.0
        };
        let nfr_avg = if fps_tx_avg > 0.0 {
            (fps_rx_avg / fps_tx_avg).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let rho = sarsa_cfg.nfr_thresh;

        // 3. RTT penalty
        let rtt_avg_s = self.rtt_average.get_average().as_secs_f32().min(1.0);
        let sigma_s = sarsa_cfg.rtt_thresh_ms / 1000.0;

        let penalty = if nfr_avg < rho || rtt_avg_s > sigma_s {
            1.0
        } else {
            0.0
        };

        let reward = bitrate_norm - penalty;

        reward
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
    ) {
        self.rtt_average.submit_sample(network_rtt);

        self.peak_throughput_average
            .submit_sample(peak_throughput_bps);

        self.frame_interarrival_average
            .submit_sample(frame_interarrival_s);
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
                BitrateMode::Sarsa {
                    update_interval_s,
                    max_bitrate_mbps,
                    min_bitrate_mbps,
                    nfr_thresh,
                    rtt_thresh_ms,
                    agent_config,
                    ..
                } => {
                    self.update_interval_s = Duration::from_secs_f32(*update_interval_s);

                    let state_dim = 5;
                    let action_values = vec![
                        -agent_config.action_step_percent / 100.0,
                        0.0,
                        agent_config.action_step_percent / 100.0,
                    ];

                    self.sarsa_agent = Some(SarsaAgent::new(SarsaAgentConfig {
                        epsilon: agent_config.epsilon,
                        gamma: agent_config.gamma,
                        lr: agent_config.lr,
                        state_dim,
                        hidden_dim: agent_config.hidden_dim as i64,
                        action_values: action_values,
                        max_bitrate_mbps: *max_bitrate_mbps,
                        min_bitrate_mbps: *min_bitrate_mbps,
                        nfr_thresh: *nfr_thresh,
                        rtt_thresh_ms: *rtt_thresh_ms,
                    }));
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
                &mut self.frame_interarrival_average,
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
                fn round_down_to_nearest_mult_from_prev(
                    value: f32,
                    r_step: f32,
                    step: f32,
                    prev: f32,
                    max: f32,
                    min: f32,
                ) -> f32 {
                    if value >= prev {
                        let steps_to_value = ((value - prev) / step).floor();
                        let steps_to_max = ((max - prev) / step).floor();
                        let n = steps_to_value.min(steps_to_max);
                        prev + n * step
                    } else {
                        let steps_to_min = ((prev - min) / r_step).floor();
                        let steps_to_value = ((prev - value) / r_step).ceil();
                        let n = steps_to_min.min(steps_to_value);
                        prev - n * r_step
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

                // Sample from uniform distribution
                let mut rng = thread_rng();
                let uniform_dist = Uniform::new(0.0, 1.0);
                let random_prob = rng.sample(uniform_dist);

                let mut bitrate_bps: f32 = self.last_target_bitrate_bps;

                let frame_interval_s = self.frame_interval_average.get_average().as_secs_f32();
                let rtt_avg_heur_s = self.rtt_average.get_average().as_secs_f32();

                let server_fps = if frame_interval_s != 0.0 {
                    1.0 / frame_interval_s
                } else {
                    0.0
                };
                let heur_fps = if self.frame_interarrival_average.get_average() != 0.0 {
                    1.0 / self.frame_interarrival_average.get_average()
                } else {
                    0.0
                };

                let estimated_capacity_bps = self.peak_throughput_average.get_average();
                let steps_bps = profile_config.step_size_mbps * 1E6;
                let r_steps_bps = profile_config.r_step_size_mbps * 1E6;

                let threshold_fps = profile_config.nfr_thresh * server_fps;
                let threshold_rtt = frame_interval_s * profile_config.rtt_thresh_scaling_factor;
                let threshold_u = profile_config.rtt_explor_prob;

                if heur_fps >= threshold_fps {
                    if rtt_avg_heur_s > threshold_rtt {
                        if random_prob >= threshold_u {
                            bitrate_bps -= r_steps_bps; // decrease bitrate by 1 step
                        }
                    } else {
                        if random_prob <= threshold_u {
                            bitrate_bps += steps_bps; // increase bitrate by 1 step
                        }
                    }
                } else {
                    bitrate_bps -= r_steps_bps; // decrease bitrate by 1 step
                }

                // Ensure bitrate is below the estimated network capacity
                let capacity_upper_limit =
                    profile_config.capacity_scaling_factor * estimated_capacity_bps;

                bitrate_bps = f32::min(bitrate_bps, capacity_upper_limit);

                // Ensure bitrate is always within the configured range
                bitrate_bps = minmax_bitrate(
                    bitrate_bps,
                    profile_config.max_bitrate_mbps * 1E6,
                    profile_config.min_bitrate_mbps * 1E6,
                );

                bitrate_bps = round_down_to_nearest_mult_from_prev(
                    bitrate_bps,
                    r_steps_bps,
                    steps_bps,
                    self.last_target_bitrate_bps,
                    profile_config.max_bitrate_mbps * 1E6,
                    profile_config.min_bitrate_mbps * 1E6,
                );

                let heur_stats = HeuristicStats {
                    frame_interval_s: frame_interval_s,
                    server_fps: server_fps, // fps_tx
                    steps_bps: steps_bps,
                    r_steps_bps: r_steps_bps,

                    network_heur_fps: heur_fps, // fps_rx
                    rtt_avg_heur_s: rtt_avg_heur_s,
                    random_prob: random_prob,

                    threshold_fps: threshold_fps,
                    threshold_rtt_s: threshold_rtt,
                    threshold_u: threshold_u,

                    requested_bitrate_bps: bitrate_bps,
                };
                alvr_events::send_event(EventType::HeuristicStats(heur_stats));

                stats.manual_max_bps = Some(profile_config.max_bitrate_mbps * 1e6);
                stats.manual_min_bps = Some(profile_config.min_bitrate_mbps * 1e6);

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
            BitrateMode::Sarsa { .. } => {
                // 1. Compute reward from last interval
                let r_prev = self.compute_reward(); // reward associated with previous (s_{t-1}, a_{t-1})

                // 2. Build current state (normalized feature vector)
                let s_t = self.build_state_vector();

                // 3. Select next action (ε-greedy)
                let agent = self
                    .sarsa_agent
                    .as_mut()
                    .expect("SARSA agent not initialized");
                let (a_t_value, a_t_idx, is_greedy) = agent.select_action(&s_t);

                // 4. If we have a stored previous transition, perform SARSA update:
                //    update_transition(s_{t-1}, a_{t-1}, r_{t-1}, s_t, a_t)
                if let Some(a_prev_idx) = agent.a_prev_idx {
                    if let Some(s_prev_tensor) = agent.s_prev.as_ref().map(|t| t.shallow_clone()) {
                        agent.update(&s_prev_tensor, a_prev_idx, r_prev, &s_t, a_t_idx);
                    }
                } else {
                    // No previous transition available (first step), skipping update this round and only store the current transition below.
                }

                // 5. Compute new bitrate
                let bitrate_bps = ((1.0 + a_t_value) * self.last_target_bitrate_bps).clamp(
                    agent.cfg.min_bitrate_mbps as f32 * 1e6,
                    agent.cfg.max_bitrate_mbps as f32 * 1e6,
                );

                // previous state
                let s_prev_vec: Option<Vec<f32>> = agent.s_prev.as_ref().map(|prev| {
                    Vec::try_from(prev.view([-1]).shallow_clone())
                        .expect("s_prev tensor must be 1D f32")
                });
                let s_prev_str = match &s_prev_vec {
                    Some(v) => format!("{:?}", v),
                    None => "[]".to_string(),
                };

                let s_t_vec: Vec<f32> = Vec::try_from(s_t.view([-1]).shallow_clone())
                    .expect("s_t tensor must be 1D f32");
                let s_t_str = format!("{:?}", s_t_vec);

                let sarsa_stats = SARSAStats {
                    s_prev: s_prev_str,                 // previous state
                    a_prev_idx: agent.a_prev_idx,       // previous action index
                    r_prev,                             // previous reward
                    s_t: s_t_str,                       // current state
                    a_t_idx,                            // current action index
                    a_t_value,                          // current action value
                    is_greedy: is_greedy,               // whether current action is greedy
                    requested_bitrate_bps: bitrate_bps, // requested bitrate
                };
                alvr_events::send_event(EventType::SARSAStats(sarsa_stats));

                // 6. Store current state and action inside the agent for the next update
                agent.s_prev = Some(s_t.shallow_clone());
                agent.a_prev_idx = Some(a_t_idx);

                bitrate_bps
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

        (
            FfiDynamicEncoderParams {
                updated: 1,
                bitrate_bps: bitrate_bps as u64,
                framerate: 1.0 / frame_interval.as_secs_f32().min(1.0),
            },
            Some(stats),
        )
    }
}
