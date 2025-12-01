use serde::{Deserialize, Serialize};
use std::net::IpAddr;

pub fn find_client_interface(
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

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct APStats {
    pub interfaces: Vec<Interface>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Interface {
    #[serde(default)]
    pub interface: String,
    #[serde(default)]
    pub mac: String,
    #[serde(default)]
    pub essid: String,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub channel: String,
    #[serde(default)]
    pub channel_ghz: String,
    #[serde(default)]
    pub ht_mode: String,
    #[serde(default)]
    pub tx_power_dbm: String,
    #[serde(default)]
    pub link_quality: String,
    #[serde(default)]
    pub signal_dbm: String,
    #[serde(default)]
    pub noise_dbm: String,
    #[serde(default)]
    pub bitrate_mbps: String,
    #[serde(default)]
    pub rx_pck_s: String,
    #[serde(default)]
    pub tx_pck_s: String,
    #[serde(default)]
    pub rx_kbytes_s: String,
    #[serde(default)]
    pub tx_kbytes_s: String,
    #[serde(default)]
    pub rx_cmp_s: String,
    #[serde(default)]
    pub tx_cmp_s: String,
    #[serde(default)]
    pub rx_mcst_s: String,
    #[serde(default)]
    pub if_util: String,
    #[serde(default)]
    pub ch_active_time_ms: String,
    #[serde(default)]
    pub ch_busy_time_ms: String,
    #[serde(default)]
    pub ch_rx_time_ms: String,
    #[serde(default)]
    pub ch_bss_rx_time_ms: String,
    #[serde(default)]
    pub ch_tx_time_ms: String,
    #[serde(default)]
    pub clients: Vec<Client>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Client {
    #[serde(default)]
    pub ip: String,
    #[serde(default)]
    pub mac: String,
    #[serde(default)]
    pub hostname: String,
    #[serde(default)]
    pub signal_dbm: String,
    #[serde(default)]
    pub noise_dbm: String,
    #[serde(default)]
    pub snr_db: String,
    #[serde(default)]
    pub last_comm_ms: String,
    #[serde(default)]
    pub current_time_ms: String,
    #[serde(default)]
    pub rx: RxStats,
    #[serde(default)]
    pub tx: TxStats,
    #[serde(default)]
    pub expected_throughput_mbps: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RxStats {
    #[serde(default)]
    pub bitrate_mbps: String,
    #[serde(default)]
    pub mcs: String,
    #[serde(default)]
    pub bandwidth_mhz: String,
    #[serde(default)]
    pub ss: String,
    #[serde(default)]
    pub packets: String,
    #[serde(default)]
    pub bytes: String,
    #[serde(default)]
    pub duration: String, // in microseconds
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TxStats {
    #[serde(default)]
    pub bitrate_mbps: String,
    #[serde(default)]
    pub mcs: String,
    #[serde(default)]
    pub bandwidth_mhz: String,
    #[serde(default)]
    pub ss: String,
    #[serde(default)]
    pub packets: String,
    #[serde(default)]
    pub bytes: String,
    #[serde(default)]
    pub retries: String,
    #[serde(default)]
    pub failed: String,
    #[serde(default)]
    pub duration: String, // in microseconds
}
