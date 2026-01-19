use alvr_common::{find_client_interface, info, APStats};
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;

#[derive(Clone, Debug)]
struct ClientHistory {
    last_tx_pkts: u64,
    last_tx_retries: u64,

    last_tx_us: u64,
    last_rx_us: u64,
    last_time_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct WifiMetrics {
    pub mcs_raw: f32,          // Raw MCS (0..11)
    pub channel_busy_pct: f32, // Channel utilization [0.0, 1.0]
    pub tx_retry_rate: f32,

    // Fairness Data
    pub my_airtime_fraction: f32, // My portion of the total VR airtime [0.0, 1.0]
    pub fairness_index: f32,      // Global Jain's Index [0.0, 1.0]
    pub active_vr_count: usize,   // Number of active VR clients (N_t)
}

impl Default for WifiMetrics {
    fn default() -> Self {
        Self {
            mcs_raw: 0.0,
            channel_busy_pct: 0.0,
            tx_retry_rate: 0.0,
            my_airtime_fraction: 0.0,
            fairness_index: 1.0,
            active_vr_count: 1, // Default to 1 (myself)
        }
    }
}

pub struct WifiStatsProcessor {
    client_ip: IpAddr,
    history: HashMap<String, ClientHistory>,
    prev_busy_time_ms: Option<f32>,
    prev_active_time_ms: Option<f32>,
    last_metrics: WifiMetrics,
}

impl WifiStatsProcessor {
    pub fn new(client_ip: IpAddr) -> Self {
        Self {
            client_ip,
            history: HashMap::new(),
            prev_busy_time_ms: None,
            prev_active_time_ms: None,
            last_metrics: WifiMetrics::default(),
        }
    }

    pub fn process(&mut self, buffer: &VecDeque<APStats>) -> WifiMetrics {
        if buffer.is_empty() {
            //info!("WifiStatsProcessor: Buffer is empty");
            return self.last_metrics;
        }

        // 1. Compute Channel Stats (MCS & Busy %)
        let (mcs_raw, busy_frac) = self.compute_channel_stats(buffer);

        // 2. Compute Fairness & Airtime
        let (my_frac, fairness_idx, count, my_retry_rate) = if let Some(latest) = buffer.back() {
            self.compute_client_metrics(latest)
        } else {
            (0.0, 1.0, 1, 0.0)
        };

        // 3. Update Cache
        self.last_metrics = WifiMetrics {
            mcs_raw,
            channel_busy_pct: busy_frac,
            tx_retry_rate: my_retry_rate,
            my_airtime_fraction: my_frac,
            fairness_index: fairness_idx,
            active_vr_count: count,
        };

        self.last_metrics
    }

    fn compute_channel_stats(&mut self, buffer: &VecDeque<APStats>) -> (f32, f32) {
        let mut mcs_sum = 0.0;
        let mut mcs_count = 0.0;
        let mut ch_busy_frac = 0.0;
        let last_idx = buffer.len().saturating_sub(1);

        for (i, ap_stats) in buffer.iter().enumerate() {
            let (iface_opt, client_opt) = find_client_interface(ap_stats, self.client_ip);

            // A. MCS Average
            if let Some(client) = client_opt {
                if let Some(mcs) = client.tx.mcs {
                    mcs_sum += mcs as f32;
                    mcs_count += 1.0;
                }
            }

            // B. Channel Busy Fraction (using cumulative counters)
            if i == last_idx {
                if let Some(iface) = iface_opt {
                    let busy = iface.ch_busy_time_ms.unwrap_or(0) as f32;
                    let active = iface.ch_active_time_ms.unwrap_or(1) as f32;

                    let prev_busy = self.prev_busy_time_ms.unwrap_or(busy);
                    let prev_active = self.prev_active_time_ms.unwrap_or(active);

                    let busy_delta = busy - prev_busy;
                    let active_delta = active - prev_active;

                    if active_delta > 0.0 {
                        ch_busy_frac = (busy_delta / active_delta).clamp(0.0, 1.0);
                    }

                    self.prev_busy_time_ms = Some(busy);
                    self.prev_active_time_ms = Some(active);
                }
            }
        }

        let mcs_avg = if mcs_count > 0.0 {
            mcs_sum / mcs_count
        } else {
            0.0
        };
        (mcs_avg, ch_busy_frac)
    }

    // Returns: (My_Airtime_Fraction, Jain_Index, N_Active_Clients, My_Retry_Rate)
    fn compute_client_metrics(&mut self, latest_ap_stats: &APStats) -> (f32, f32, usize, f32) {
        let (iface_opt, _) = find_client_interface(latest_ap_stats, self.client_ip);
        let iface = match iface_opt {
            Some(i) => i,
            None => return (0.0, 1.0, 1, 0.0),
        };

        let mut usage_map: HashMap<String, f32> = HashMap::new();
        let mut total_vr_airtime = 0.0;
        let mut my_retry_rate = 0.0;

        // A. Calculate raw usage for ALL VR clients
        for c in &iface.clients {
            if !c.is_vr.unwrap_or(false) {
                continue;
            }

            let (usage_opt, retry_rate_opt) = self.update_client_stats(
                &c.ip,
                c.tx.duration.unwrap_or(0),
                c.rx.duration.unwrap_or(0),
                c.current_time_ms.unwrap_or(0),
                c.tx.packets.unwrap_or(0),
                c.tx.retries.unwrap_or(0),
            );

            let usage = usage_opt.unwrap_or(0.0);
            let retry_rate = retry_rate_opt.unwrap_or(0.0);

            usage_map.insert(c.ip.clone(), usage);
            total_vr_airtime += usage;

            if c.ip == self.client_ip.to_string() {
                my_retry_rate = retry_rate; // only for self
            }
        }

        let n_clients = usage_map.len().max(1);

        // B. Calculate My Fraction (alpha_i)
        // alpha_i = My_Raw / Total_VR_Raw
        let my_raw = *usage_map.get(&self.client_ip.to_string()).unwrap_or(&0.0);
        let my_fraction = if total_vr_airtime > 0.0 {
            my_raw / total_vr_airtime
        } else {
            0.0
        };

        // C. Calculate Jain's Index
        // We use the fractions (normalized values) for Jain's index
        let mut sum_sq = 0.0;
        for raw in usage_map.values() {
            let frac = if total_vr_airtime > 0.0 {
                raw / total_vr_airtime
            } else {
                0.0
            };
            sum_sq += frac.powi(2);
        }

        let jain = if sum_sq > 0.0 {
            1.0 / (n_clients as f32 * sum_sq)
        } else {
            1.0
        };

        (my_fraction, jain, n_clients, my_retry_rate)
    }

    // Generic helper for any client IP
    fn update_client_stats(
        &mut self,
        ip: &str,
        tx: u64,
        rx: u64,
        now: u64,
        tx_pkts: u64,
        tx_retries: u64,
    ) -> (Option<f32>, Option<f32>) {
        let entry = self.history.entry(ip.to_string()).or_insert(ClientHistory {
            last_tx_us: tx,
            last_rx_us: rx,
            last_time_ms: now,
            last_tx_pkts: tx_pkts,
            last_tx_retries: tx_retries,
        });

        if tx < entry.last_tx_us
            || rx < entry.last_rx_us
            || now <= entry.last_time_ms
            || tx_pkts < entry.last_tx_pkts
            || tx_retries < entry.last_tx_retries
        {
            entry.last_tx_us = tx;
            entry.last_rx_us = rx;
            entry.last_time_ms = now;
            entry.last_tx_pkts = tx_pkts;
            entry.last_tx_retries = tx_retries;
            return (None, None);
        }

        // 1. Airtime usage
        let d_tx = tx - entry.last_tx_us;
        let d_rx = rx - entry.last_rx_us;
        let d_time_us = (now - entry.last_time_ms) * 1000;

        let airtime_usage = if d_time_us > 0 {
            (d_tx + d_rx) as f32 / d_time_us as f32
        } else {
            0.0
        };

        // 2. Retry rate
        let d_tx_pkts = tx_pkts - entry.last_tx_pkts;
        let d_tx_retries = tx_retries - entry.last_tx_retries;
        let d_tx_total = d_tx_pkts + d_tx_retries;

        let retry_rate = if d_tx_total > 0 {
            d_tx_retries as f32 / d_tx_total as f32
        } else {
            0.0
        };

        entry.last_tx_us = tx;
        entry.last_rx_us = rx;
        entry.last_time_ms = now;
        entry.last_tx_pkts = tx_pkts;
        entry.last_tx_retries = tx_retries;

        (Some(airtime_usage), Some(retry_rate))
    }
}
