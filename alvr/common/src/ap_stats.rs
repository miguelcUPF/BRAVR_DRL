use log::warn;
use serde::{Deserialize, Deserializer, Serialize};
use std::net::IpAddr;

pub fn find_client_interface<'a>(
    ap_stats: &'a APStats,
    client_ip: IpAddr,
) -> (Option<&'a Interface>, Option<&'a Client>) {
    for iface in &ap_stats.interfaces {
        for client in &iface.clients {
            if client.ip_addr() == Some(client_ip) {
                return (Some(iface), Some(client));
            }
        }
    }
    (None, None)
}

pub fn de_opt_u64_any<'de, D>(d: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum U64OrStr {
        Int(u64),
        Float(f64),
        Str(String),
    }

    match Option::<U64OrStr>::deserialize(d)? {
        None => Ok(None),
        Some(U64OrStr::Int(v)) => Ok(Some(v)),
        Some(U64OrStr::Float(f)) => Ok(Some(f as u64)), // truncate float
        Some(U64OrStr::Str(s)) => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("N/A") || s.eq_ignore_ascii_case("unknown") {
                return Ok(None);
            }
            match s.parse::<f64>() {
                Ok(f) => Ok(Some(f as u64)),
                Err(e) => {
                    warn!("Failed to parse u64 from '{}': {}", s, e);
                    Ok(None)
                }
            }
        }
    }
}

/// Deserialize i16 from string, int, or float
pub fn de_opt_i16_any<'de, D>(d: D) -> Result<Option<i16>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum I16OrStr {
        Int(i16),
        Float(f64),
        Str(String),
    }

    match Option::<I16OrStr>::deserialize(d)? {
        None => Ok(None),
        Some(I16OrStr::Int(v)) => Ok(Some(v)),
        Some(I16OrStr::Float(f)) => Ok(Some(f as i16)),
        Some(I16OrStr::Str(s)) => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("N/A") || s.eq_ignore_ascii_case("unknown") {
                return Ok(None);
            }
            match s.parse::<f64>() {
                Ok(f) => Ok(Some(f as i16)),
                Err(e) => {
                    warn!("Failed to parse i16 from '{}': {}", s, e);
                    Ok(None)
                }
            }
        }
    }
}

/// Deserialize f32 from string, int, or float
pub fn de_opt_f32_any<'de, D>(d: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum F32OrStr {
        Float(f64),
        Int(i64),
        Str(String),
    }

    match Option::<F32OrStr>::deserialize(d)? {
        None => Ok(None),
        Some(F32OrStr::Float(f)) => Ok(Some(f as f32)),
        Some(F32OrStr::Int(i)) => Ok(Some(i as f32)),
        Some(F32OrStr::Str(s)) => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("N/A") || s.eq_ignore_ascii_case("unknown") {
                return Ok(None);
            }
            match s.parse::<f32>() {
                Ok(f) => Ok(Some(f)),
                Err(e) => {
                    warn!("Failed to parse f32 from '{}': {}", s, e);
                    Ok(None)
                }
            }
        }
    }
}

/// Deserialize bool from string, int, or boolean
pub fn de_opt_bool_any<'de, D>(d: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrStr {
        Bool(bool),
        Int(u64),
        Str(String),
    }

    match Option::<BoolOrStr>::deserialize(d)? {
        None => Ok(None),
        Some(BoolOrStr::Bool(b)) => Ok(Some(b)),
        Some(BoolOrStr::Int(1)) => Ok(Some(true)),
        Some(BoolOrStr::Int(0)) => Ok(Some(false)),
        Some(BoolOrStr::Int(_)) => Ok(None),
        Some(BoolOrStr::Str(s)) => {
            let s = s.trim();
            match s.to_ascii_lowercase().as_str() {
                "true" | "1" => Ok(Some(true)),
                "false" | "0" => Ok(Some(false)),
                _ => {
                    warn!("Failed to parse bool from '{}'", s);
                    Ok(None)
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct APStats {
    #[serde(default)]
    pub interfaces: Vec<Interface>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Interface {
    pub interface: String,

    #[serde(default)]
    pub mac: Option<String>,
    #[serde(default)]
    pub essid: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub ht_mode: Option<String>,

    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub channel: Option<u64>,

    #[serde(default)]
    pub channel_ghz: Option<String>,

    #[serde(default, deserialize_with = "de_opt_i16_any")]
    pub tx_power_dbm: Option<i16>,

    #[serde(default)]
    pub link_quality: Option<String>,

    #[serde(default, deserialize_with = "de_opt_i16_any")]
    pub signal_dbm: Option<i16>,
    #[serde(default, deserialize_with = "de_opt_i16_any")]
    pub noise_dbm: Option<i16>,

    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub bitrate_mbps: Option<u64>,

    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub rx_pck_s: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub tx_pck_s: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub rx_kbytes_s: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub tx_kbytes_s: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub rx_cmp_s: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub tx_cmp_s: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub rx_mcst_s: Option<f32>,

    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub if_util: Option<f32>,

    #[serde(deserialize_with = "de_opt_u64_any")]
    pub ch_active_time_ms: Option<u64>,
    #[serde(deserialize_with = "de_opt_u64_any")]
    pub ch_busy_time_ms: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub ch_rx_time_ms: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub ch_bss_rx_time_ms: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub ch_tx_time_ms: Option<u64>,

    #[serde(default)]
    pub clients: Vec<Client>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Client {
    pub ip: String,
    pub mac: String,

    #[serde(default)]
    pub hostname: Option<String>,

    #[serde(deserialize_with = "de_opt_i16_any")]
    pub signal_dbm: Option<i16>,

    #[serde(default, deserialize_with = "de_opt_i16_any")]
    pub noise_dbm: Option<i16>,

    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub snr_db: Option<u64>,

    #[serde(deserialize_with = "de_opt_bool_any")]
    pub is_vr: Option<bool>,

    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub last_comm_ms: Option<u64>,

    #[serde(deserialize_with = "de_opt_u64_any")]
    pub current_time_ms: Option<u64>,

    #[serde(default)]
    pub rx: RxStats,

    #[serde(default)]
    pub tx: TxStats,

    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub expected_throughput_mbps: Option<u64>,
}

impl Client {
    pub fn ip_addr(&self) -> Option<IpAddr> {
        self.ip.parse().ok()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct RxStats {
    #[serde(default, deserialize_with = "de_opt_f32_any")]
    pub bitrate_mbps: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub mcs: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub bandwidth_mhz: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub ss: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub packets: Option<u64>,
    #[serde(deserialize_with = "de_opt_u64_any")]
    pub bytes: Option<u64>,
    #[serde(deserialize_with = "de_opt_u64_any")]
    pub duration: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TxStats {
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub bitrate_mbps: Option<u64>,
    #[serde(deserialize_with = "de_opt_u64_any")]
    pub mcs: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub bandwidth_mhz: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub ss: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub packets: Option<u64>,
    #[serde(deserialize_with = "de_opt_u64_any")]
    pub bytes: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub retries: Option<u64>,
    #[serde(default, deserialize_with = "de_opt_u64_any")]
    pub failed: Option<u64>,
    #[serde(deserialize_with = "de_opt_u64_any")]
    pub duration: Option<u64>,
}
