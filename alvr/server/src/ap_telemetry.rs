use alvr_common::{find_client_interface, info, APStats};
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;

#[derive(Clone, Debug)]
struct ClientAirtimeInfo {
    last_tx_us: u64,
    last_rx_us: u64,
    last_time_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct WifiMetrics {
    pub mcs_raw: f32,          // Raw MCS (0..11)
    pub channel_busy_pct: f32, // Channel utilization [0.0, 1.0]

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
            my_airtime_fraction: 0.0,
            fairness_index: 1.0,
            active_vr_count: 1, // Default to 1 (myself)
        }
    }
}

pub struct WifiStatsProcessor {
    client_ip: IpAddr,
    airtime_history: HashMap<String, ClientAirtimeInfo>,
    prev_busy_time_ms: Option<f32>,
    prev_active_time_ms: Option<f32>,
    last_metrics: WifiMetrics,
}

impl WifiStatsProcessor {
    pub fn new(client_ip: IpAddr) -> Self {
        Self {
            client_ip,
            airtime_history: HashMap::new(),
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
        // We do this in one pass to ensure consistency between MyShare and Jain's Index
        let (my_frac, fairness_idx, count) = if let Some(latest) = buffer.back() {
            self.compute_fairness_and_usage(latest)
        } else {
            (0.0, 1.0, 1)
        };

        // 3. Update Cache
        self.last_metrics = WifiMetrics {
            mcs_raw,
            channel_busy_pct: busy_frac,
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

    // Returns: (My_Airtime_Fraction, Jain_Index, N_Active_Clients)
    fn compute_fairness_and_usage(&mut self, latest_ap_stats: &APStats) -> (f32, f32, usize) {
        let (iface_opt, _) = find_client_interface(latest_ap_stats, self.client_ip);
        let iface = match iface_opt {
            Some(i) => i,
            None => return (0.0, 1.0, 1),
        };

        let mut usage_map: HashMap<String, f32> = HashMap::new();
        let mut total_vr_airtime = 0.0;

        // A. Calculate raw usage for ALL VR clients
        for c in &iface.clients {
            if !c.is_vr.unwrap_or(false) {
                continue;
            }

            let raw_usage = self
                .calculate_client_raw_usage(
                    &c.ip,
                    c.tx.duration.unwrap_or(0),
                    c.rx.duration.unwrap_or(0),
                    c.current_time_ms.unwrap_or(0),
                )
                .unwrap_or(0.0);

            usage_map.insert(c.ip.clone(), raw_usage);
            total_vr_airtime += raw_usage;
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

        (my_fraction, jain, n_clients)
    }

    // Generic helper for any client IP
    fn calculate_client_raw_usage(&mut self, ip: &str, tx: u64, rx: u64, now: u64) -> Option<f32> {
        let entry = self
            .airtime_history
            .entry(ip.to_string())
            .or_insert(ClientAirtimeInfo {
                last_tx_us: tx,
                last_rx_us: rx,
                last_time_ms: now,
            });

        if tx < entry.last_tx_us || rx < entry.last_rx_us || now <= entry.last_time_ms {
            entry.last_tx_us = tx;
            entry.last_rx_us = rx;
            entry.last_time_ms = now;
            return None;
        }

        let usage = (tx - entry.last_tx_us) + (rx - entry.last_rx_us);
        let duration = (now - entry.last_time_ms) * 1000;

        entry.last_tx_us = tx;
        entry.last_rx_us = rx;
        entry.last_time_ms = now;

        // Return raw utilization (0.0 to 1.0)
        Some(usage as f32 / duration as f32)
    }
}
