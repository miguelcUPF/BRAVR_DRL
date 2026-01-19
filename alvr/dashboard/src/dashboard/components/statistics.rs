use crate::{dashboard::theme::graph_colors, dashboard::ServerRequest};
use alvr_common::{find_client_interface, APStats, Client, Interface};
use alvr_events::{GraphNetworkStatistics, GraphStatistics, SARSAStats, StatisticsSummary};
use alvr_gui_common::theme;
use eframe::{
    egui::{
        popup, pos2, vec2, Align2, Color32, FontId, Frame, Id, Painter, Rect, RichText, Rounding,
        ScrollArea, Shape, Stroke, Ui,
    },
    emath::RectTransform,
    epaint::Pos2,
};
use statrs::statistics::{self, OrderStatistics};
use std::{collections::VecDeque, net::IpAddr, ops::RangeInclusive};

const GRAPH_HISTORY_SIZE: usize = 1000;
const GRAPH_HISTORY_SIZE_SARSA: usize = 100;
const GRAPH_HISTORY_SIZE_AP: usize = 100;
const UPPER_QUANTILE: f64 = 0.80;
// const LOWER_QUANTILE: f64 = 0.2;
// const MIDDLE_QUANTILE: f64 = 0.5;
fn draw_lines(painter: &Painter, points: Vec<Pos2>, color: Color32) {
    painter.add(Shape::line(points, Stroke::new(1.0, color)));
}

fn series_color(name: &str) -> Color32 {
    use fxhash::hash64;
    let h = hash64(name.as_bytes());
    let min_val = 100; // ensures visible brightness
    let r = (((h >> 16) & 0xFF) as u8).max(min_val);
    let g = (((h >> 8) & 0xFF) as u8).max(min_val);
    let b = ((h & 0xFF) as u8).max(min_val);

    Color32::from_rgb(r, g, b)
}
// SERIES GRAPH MACRO FOR AP STATS
macro_rules! make_ap_series_graph {
    (
        fn $fn_name:ident ($self_ident:ident, $ui_ident:ident, $width_ident:ident),
        title = $title:expr,
        yrange = $yrange:expr,
        series = { $( $label:expr => $closure:expr ),+ $(,)? },
        tooltip = |$tui_ident:ident, $stats_ident:ident| $tooltip_block:block
    ) => {

        fn $fn_name(&$self_ident, $ui_ident: &mut Ui, $width_ident: f32) {
            if $self_ident.history_ap.is_empty() {
                return;
            }

            let Some(client_ip) = $self_ident.client_ip else { return };

            $self_ident.draw_ap_graph(
                $ui_ident,
                $width_ident,
                $title,
                $yrange,

                move |painter, to_screen_trans| {
                    $(
                        let mut pts: Vec<Pos2> = Vec::with_capacity(GRAPH_HISTORY_SIZE_AP);

                        for i in 0..GRAPH_HISTORY_SIZE_AP {
                            let ap_stats = &$self_ident.history_ap[i];
                            let (iface_opt, client_opt) = find_client_interface(ap_stats, client_ip);
                            let iface = match iface_opt {
                                Some(v) => v,
                                None => { pts.push(to_screen_trans * pos2(i as f32, 0.0)); continue; }
                            };

                            let v = ($closure)(&iface, client_opt);
                            pts.push(to_screen_trans * pos2(i as f32, v as f32));
                        }

                        // deterministic per-series color
                        let color = series_color($label);

                        draw_lines(painter, pts, color);
                    )+
                },

                move |$tui_ident: &mut Ui, $stats_ident: &APStats| {
                    let (iface_opt, _client_opt) = find_client_interface($stats_ident, client_ip);
                    if iface_opt.is_none() {
                        $tui_ident.label("Interface not found");
                        return;
                    }

                    $tooltip_block
                }
            );
        }
    }
}

// CUSTOM GRAPH MACRO FOR AP STATS (rectangles, stacked bars, etc.)
macro_rules! make_ap_custom_graph {
    (
        fn $fn_name:ident ($self_ident:ident, $ui_ident:ident, $width_ident:ident),
        title = $title:expr,
        yrange = $yrange:expr,
        paint = $paint:expr,
        tooltip = |$tui_ident:ident, $stats_ident:ident| $tooltip_block:block
    ) => {
        fn $fn_name(&$self_ident, $ui_ident: &mut Ui, $width_ident: f32) {
            if $self_ident.history_ap.is_empty() {
                return;
            }

            let Some(client_ip) = $self_ident.client_ip else {
                return;
            };

            $self_ident.draw_ap_graph(
                $ui_ident,
                $width_ident,
                $title,
                $yrange,

                move |painter, to_screen_trans| {
                    for i in 0..GRAPH_HISTORY_SIZE_AP {
                        let ap_stats = &$self_ident.history_ap[i];
                        let (iface_opt, client_opt) = find_client_interface(ap_stats, client_ip);
                        let iface = match iface_opt { Some(v) => v, None => continue };
                        let clients = &iface.clients;

                        ($paint)(painter, to_screen_trans, i, &iface, clients, client_opt.as_ref());
                    }
                },

                move |$tui_ident: &mut Ui, $stats_ident: &APStats| {
                    let (iface_opt, _) = find_client_interface($stats_ident, client_ip);
                    if iface_opt.is_none() {
                        $tui_ident.label("Interface not found");
                        return;
                    }
                    $tooltip_block
                }
            );
        }
    };
}

pub struct StatisticsTab {
    history: VecDeque<GraphStatistics>,
    history_network: VecDeque<GraphNetworkStatistics>,
    history_sarsa: VecDeque<SARSAStats>,
    history_ap: VecDeque<APStats>,
    last_statistics_summary: Option<StatisticsSummary>,
    client_ip: Option<IpAddr>,
    bulk_ap_stats: bool,
    ap_stats_enabled: bool,
    sarsa_stats_enabled: bool,
}

impl StatisticsTab {
    pub fn new() -> Self {
        Self {
            history: vec![GraphStatistics::default(); GRAPH_HISTORY_SIZE]
                .into_iter()
                .collect(),
            history_network: vec![GraphNetworkStatistics::default(); GRAPH_HISTORY_SIZE]
                .into_iter()
                .collect(),
            history_sarsa: vec![SARSAStats::default(); GRAPH_HISTORY_SIZE_SARSA]
                .into_iter()
                .collect(),
            history_ap: vec![APStats::default(); GRAPH_HISTORY_SIZE_AP]
                .into_iter()
                .collect(),
            last_statistics_summary: None,
            client_ip: None,
            bulk_ap_stats: false,
            ap_stats_enabled: false,
            sarsa_stats_enabled: false,
        }
    }

    pub fn update_client_ip(&mut self, client_ip: IpAddr) {
        self.client_ip = Some(client_ip);
    }

    pub fn enable_ap_stats(&mut self) {
        self.ap_stats_enabled = true;
    }

    pub fn disable_ap_stats(&mut self) {
        self.ap_stats_enabled = false;
    }

    pub fn enable_bulk_ap_stats(&mut self) {
        self.bulk_ap_stats = true;
    }

    pub fn disable_bulk_ap_stats(&mut self) {
        self.bulk_ap_stats = false;
    }

    pub fn enable_sarsa_stats(&mut self) {
        self.sarsa_stats_enabled = true;
    }

    pub fn disable_sarsa_stats(&mut self) {
        self.sarsa_stats_enabled = false;
    }

    pub fn update_statistics(&mut self, statistics: StatisticsSummary) {
        self.last_statistics_summary = Some(statistics);
    }

    pub fn update_graph_statistics(&mut self, statistics: GraphStatistics) {
        self.history.pop_front();
        self.history.push_back(statistics);
    }

    pub fn update_graph_network_statistics(&mut self, statistics: GraphNetworkStatistics) {
        self.history_network.pop_front();
        self.history_network.push_back(statistics);
    }

    pub fn update_sarsa_stats(&mut self, statistics: SARSAStats) {
        self.history_sarsa.pop_front();
        self.history_sarsa.push_back(statistics);
    }

    pub fn update_ap_stats(&mut self, statistics: APStats) {
        self.history_ap.pop_front();
        self.history_ap.push_back(statistics);
    }

    pub fn draw_bulk_ap_graphs(&self, ui: &mut Ui, width: f32) {
        if !self.bulk_ap_stats {
            return;
        }
        self.draw_ap_client_snr_graph(ui, width);
        self.draw_ap_client_tx_rx_mcs_graph(ui, width);
        self.draw_ap_interface_bytes_mbps_graph(ui, width);
        self.draw_ap_interface_quality_graph(ui, width);
    }

    pub fn draw_ap_info_message(&self, ui: &mut Ui) {
        if self.history_ap.is_empty() {
            return;
        }

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("ℹ TX and RX are described from the access point's interface perspective. TX = AP -> client, RX = client -> AP")
                    .size(12.0)
                    .color(Color32::GRAY)
            );
        });
    }

    pub fn ui(&mut self, ui: &mut Ui) -> Option<ServerRequest> {
        if let Some(stats) = &self.last_statistics_summary {
            ScrollArea::new([false, true]).show(ui, |ui| {
                let available_width = ui.available_width();
                self.draw_latency_graph(ui, available_width);
                self.draw_fps_graph(ui, available_width);
                self.draw_bitrate_graph(ui, available_width);
                self.draw_throughput_graphs(ui, available_width);
                self.draw_jitter(ui, available_width);
                self.draw_frameloss(ui, available_width);
                self.draw_frame_span_interarrival(ui, available_width);
                if self.sarsa_stats_enabled {
                    ui.separator();
                    self.draw_sarsa_rewards(ui, available_width);
                    self.draw_sarsa_reward_components(ui, available_width);
                    self.draw_sarsa_td_error(ui, available_width);
                    self.draw_sarsa_entropy(ui, available_width);
                    self.draw_sarsa_q_values(ui, available_width);
                    self.draw_sarsa_action_probs(ui, available_width);
                }
                ui.separator();
                if self.ap_stats_enabled {
                    self.draw_ap_clients_tx_mcs_graph(ui, available_width);
                    if self.bulk_ap_stats {
                        self.draw_ap_clients_rx_mcs_graph(ui, available_width);
                    }
                    self.draw_ap_clients_airtime_graph(ui, available_width);
                    self.draw_ap_interface_channel_activity_graph(ui, available_width);
                    self.draw_ap_clients_count_graph(ui, available_width);
                    self.draw_bulk_ap_graphs(ui, available_width);
                    self.draw_ap_info_message(ui);
                    ui.separator();
                }
                self.draw_statistics_overview(ui, stats);
            });
        } else {
            ui.heading("No statistics available");
        }

        None
    }

    fn draw_graph(
        &self,
        ui: &mut Ui,
        available_width: f32,
        title: &str,
        data_range: RangeInclusive<f32>,
        graph_content: impl FnOnce(&Painter, RectTransform),
        tooltip_content: impl FnOnce(&mut Ui, &GraphStatistics),
    ) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(20.0));

        let canvas_response = Frame::canvas(ui.style()).show(ui, |ui| {
            ui.ctx().request_repaint();
            let size = available_width * vec2(1.0, 0.2);

            let (_id, canvas_rect) = ui.allocate_space(size);

            let max = *data_range.end();
            let min = *data_range.start();
            let data_rect = Rect::from_x_y_ranges(0.0..=GRAPH_HISTORY_SIZE as f32, max..=min);
            let to_screen = RectTransform::from_to(data_rect, canvas_rect);

            let painter = ui.painter().with_clip_rect(canvas_rect);

            graph_content(&painter, to_screen);

            ui.painter().text(
                to_screen * pos2(0.0, min),
                Align2::LEFT_BOTTOM,
                format!("{:.0}", min),
                FontId::monospace(12.0),
                Color32::GRAY,
            );
            ui.painter().text(
                to_screen * pos2(0.0, max),
                Align2::LEFT_TOP,
                format!("{:.0}", max),
                FontId::monospace(12.0),
                Color32::GRAY,
            );

            data_rect
        });

        if let Some(pos) = canvas_response.response.hover_pos() {
            let graph_pos =
                RectTransform::from_to(canvas_response.response.rect, canvas_response.inner) * pos;
            let history_index = (graph_pos.x as usize).clamp(0, GRAPH_HISTORY_SIZE - 1);

            popup::show_tooltip(ui.ctx(), Id::new("popup"), |ui| {
                tooltip_content(ui, self.history.get(history_index).unwrap())
            });
        }
    }

    fn draw_sarsa_graph(
        &self,
        ui: &mut Ui,
        available_width: f32,
        title: &str,
        data_range: RangeInclusive<f32>,
        graph_content: impl FnOnce(&Painter, RectTransform),
        tooltip_content: impl FnOnce(&mut Ui, &SARSAStats),
    ) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(20.0));

        let canvas_response = Frame::canvas(ui.style()).show(ui, |ui| {
            ui.ctx().request_repaint();
            let size = available_width * vec2(1.0, 0.2);
            let (_id, canvas_rect) = ui.allocate_space(size);

            let max = *data_range.end();
            let min = *data_range.start();

            let data_rect = Rect::from_x_y_ranges(0.0..=GRAPH_HISTORY_SIZE_SARSA as f32, max..=min);
            let to_screen = RectTransform::from_to(data_rect, canvas_rect);

            let painter = ui.painter().with_clip_rect(canvas_rect);

            graph_content(&painter, to_screen);

            ui.painter().text(
                to_screen * pos2(0.0, min),
                Align2::LEFT_BOTTOM,
                format!("{:.2}", min),
                FontId::monospace(12.0),
                Color32::GRAY,
            );
            ui.painter().text(
                to_screen * pos2(0.0, max),
                Align2::LEFT_TOP,
                format!("{:.2}", max),
                FontId::monospace(12.0),
                Color32::GRAY,
            );

            data_rect
        });

        if let Some(pos) = canvas_response.response.hover_pos() {
            let graph_pos =
                RectTransform::from_to(canvas_response.response.rect, canvas_response.inner) * pos;
            let history_index = (graph_pos.x as usize).clamp(0, GRAPH_HISTORY_SIZE_SARSA - 1);

            if let Some(stats) = self.history_sarsa.get(history_index) {
                popup::show_tooltip(ui.ctx(), Id::new("sarsa_popup"), |ui| {
                    tooltip_content(ui, stats)
                });
            }
        }
    }

    fn draw_ap_graph(
        &self,
        ui: &mut Ui,
        available_width: f32,
        title: &str,
        data_range: RangeInclusive<f32>,
        graph_content: impl FnOnce(&Painter, RectTransform),
        tooltip_content: impl FnOnce(&mut Ui, &APStats),
    ) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(20.0));

        let canvas_response = Frame::canvas(ui.style()).show(ui, |ui| {
            ui.ctx().request_repaint();
            let size = available_width * vec2(1.0, 0.2);

            let (_id, canvas_rect) = ui.allocate_space(size);

            let max = *data_range.end();
            let min = *data_range.start();
            let data_rect = Rect::from_x_y_ranges(0.0..=GRAPH_HISTORY_SIZE_AP as f32, max..=min);
            let to_screen = RectTransform::from_to(data_rect, canvas_rect);

            let painter = ui.painter().with_clip_rect(canvas_rect);

            graph_content(&painter, to_screen);

            ui.painter().text(
                to_screen * pos2(0.0, min),
                Align2::LEFT_BOTTOM,
                format!("{:.0}", min),
                FontId::monospace(12.0),
                Color32::GRAY,
            );
            ui.painter().text(
                to_screen * pos2(0.0, max),
                Align2::LEFT_TOP,
                format!("{:.0}", max),
                FontId::monospace(12.0),
                Color32::GRAY,
            );

            data_rect
        });

        if let Some(pos) = canvas_response.response.hover_pos() {
            let graph_pos =
                RectTransform::from_to(canvas_response.response.rect, canvas_response.inner) * pos;
            let history_index = (graph_pos.x as usize).clamp(0, GRAPH_HISTORY_SIZE_AP - 1);

            if let Some(stats) = self.history_ap.get(history_index) {
                popup::show_tooltip(ui.ctx(), Id::new("ap_popup"), |ui| {
                    tooltip_content(ui, stats)
                });
            }
        }
    }

    fn draw_sarsa_td_error(&self, ui: &mut Ui, available_width: f32) {
        if self.history_sarsa.is_empty() {
            return;
        }

        let mut data = statistics::Data::new(
            self.history_sarsa
                .iter()
                .map(|s| s.td_error.abs() as f64)
                .collect::<Vec<_>>(),
        );

        let lower = 0.0;
        let upper = data.quantile(0.95) as f32;

        self.draw_sarsa_graph(
            ui,
            available_width,
            "SARSA |TD Error|",
            lower..=upper,
            |painter, to_screen_trans| {
                let mut points = Vec::with_capacity(GRAPH_HISTORY_SIZE_SARSA);

                for i in 0..GRAPH_HISTORY_SIZE_SARSA {
                    let stats = &self.history_sarsa[i];
                    points.push(to_screen_trans * pos2(i as f32, stats.td_error.abs()));
                }

                draw_lines(painter, points, Color32::RED);
            },
            |ui, stats| {
                ui.colored_label(
                    Color32::RED,
                    format!("|TD error|: {:.4}", stats.td_error.abs()),
                );
                ui.label(format!("Raw TD error: {:.4}", stats.td_error));
            },
        );
    }

    fn draw_sarsa_entropy(&self, ui: &mut Ui, available_width: f32) {
        if self.history_sarsa.is_empty() {
            return;
        }

        let mut data = statistics::Data::new(
            self.history_sarsa
                .iter()
                .map(|s| s.policy_entropy as f64)
                .collect::<Vec<_>>(),
        );

        let lower = 0.0;
        let upper = data.quantile(1.0) as f32;

        self.draw_sarsa_graph(
            ui,
            available_width,
            "SARSA Policy Entropy",
            lower..=upper,
            |painter, to_screen_trans| {
                let mut points = Vec::with_capacity(GRAPH_HISTORY_SIZE_SARSA);

                for i in 0..GRAPH_HISTORY_SIZE_SARSA {
                    let stats = &self.history_sarsa[i];
                    points.push(to_screen_trans * pos2(i as f32, stats.policy_entropy));
                }

                draw_lines(painter, points, Color32::LIGHT_BLUE);
            },
            |ui, stats| {
                ui.colored_label(
                    Color32::LIGHT_BLUE,
                    format!("Entropy: {:.3}", stats.policy_entropy),
                );
            },
        );
    }

    fn draw_sarsa_q_values(&self, ui: &mut Ui, available_width: f32) {
        if self.history_sarsa.is_empty() {
            return;
        }

        let mut all_qs = Vec::new();
        for s in &self.history_sarsa {
            for q in &s.q_values {
                all_qs.push(*q as f64);
            }
        }

        let mut data = statistics::Data::new(all_qs);
        let lower = data.quantile(0.05) as f32;
        let upper = data.quantile(0.95) as f32;

        let num_actions = self.history_sarsa[0].q_values.len();

        self.draw_sarsa_graph(
            ui,
            available_width,
            "SARSA Q-values",
            lower..=upper,
            |painter, to_screen_trans| {
                for a in 0..num_actions {
                    let mut points = Vec::with_capacity(GRAPH_HISTORY_SIZE_SARSA);

                    for i in 0..GRAPH_HISTORY_SIZE_SARSA {
                        let stats = &self.history_sarsa[i];
                        points.push(to_screen_trans * pos2(i as f32, stats.q_values[a]));
                    }

                    let color = Color32::from_rgb(
                        (200 + a as u8 * 30) % 255,
                        (100 + a as u8 * 60) % 255,
                        (50 + a as u8 * 90) % 255,
                    );

                    draw_lines(painter, points, color);
                }
            },
            |ui, stats| {
                ui.label("Q-values:");
                for (i, q) in stats.q_values.iter().enumerate() {
                    let color = Color32::from_rgb(
                        (200 + i as u8 * 30) % 255,
                        (100 + i as u8 * 60) % 255,
                        (50 + i as u8 * 90) % 255,
                    );
                    ui.colored_label(color, format!("Q[a{}]: {:.3}", i, q));
                }
            },
        );
    }

    fn draw_sarsa_action_probs(&self, ui: &mut Ui, available_width: f32) {
        if self.history_sarsa.is_empty() {
            return;
        }

        let num_actions = self.history_sarsa[0].action_probs.len();

        self.draw_sarsa_graph(
            ui,
            available_width,
            "SARSA Action Probabilities",
            0.0..=1.0,
            |painter, to_screen_trans| {
                for a in 0..num_actions {
                    let mut points = Vec::with_capacity(GRAPH_HISTORY_SIZE_SARSA);

                    for i in 0..GRAPH_HISTORY_SIZE_SARSA {
                        let stats = &self.history_sarsa[i];
                        let p = stats.action_probs[a];
                        points.push(to_screen_trans * pos2(i as f32, p));
                    }

                    let color = Color32::from_rgb(
                        (200 + a as u8 * 30) % 255,
                        (100 + a as u8 * 60) % 255,
                        (50 + a as u8 * 90) % 255,
                    );

                    draw_lines(painter, points, color);
                }
            },
            |ui, stats| {
                ui.label("Action probabilities:");
                for (i, p) in stats.action_probs.iter().enumerate() {
                    let color = Color32::from_rgb(
                        (200 + i as u8 * 30) % 255,
                        (100 + i as u8 * 60) % 255,
                        (50 + i as u8 * 90) % 255,
                    );
                    ui.colored_label(color, format!("a{}: {:.3}", i, p));
                }
            },
        );
    }

    fn draw_sarsa_rewards(&self, ui: &mut Ui, available_width: f32) {
        if self.history_sarsa.is_empty() {
            return;
        }

        let mut data = statistics::Data::new(
            self.history_sarsa
                .iter()
                .map(|s| s.r_prev as f64)
                .collect::<Vec<_>>(),
        );

        let lower = data.quantile(0.05) as f32;
        let upper = data.quantile(1.0) as f32;

        self.draw_sarsa_graph(
            ui,
            available_width,
            "SARSA Rewards",
            lower..=upper,
            |painter, to_screen_trans| {
                let mut reward_points = Vec::with_capacity(GRAPH_HISTORY_SIZE_SARSA);

                for i in 0..GRAPH_HISTORY_SIZE_SARSA {
                    let stats = &self.history_sarsa[i];
                    reward_points.push(to_screen_trans * pos2(i as f32, stats.r_prev));
                }

                draw_lines(painter, reward_points, Color32::GREEN);
            },
            |ui, stats| {
                ui.colored_label(Color32::GREEN, format!("Reward: {:.2}", stats.r_prev));
                ui.label(format!(
                    "Bitrate: {:.1} Mbps",
                    stats.requested_bitrate_bps / 1e6
                ));
                ui.label(format!("Action idx: {}", stats.a_t_idx));

                if stats.matches_argmax {
                    ui.label("Type: Exploit"); // sampled action equals highest Q-value
                } else {
                    ui.label("Type: Explore"); // sampled action differs from argmax
                }
            },
        );
    }

    fn draw_sarsa_reward_components(&self, ui: &mut Ui, available_width: f32) {
        if self.history_sarsa.is_empty() {
            return;
        }

        let num_components = self.history_sarsa[0].r_components.len();

        // Predefined colors for each reward component
        let colors: Vec<Color32> = vec![
            Color32::LIGHT_GREEN,  // bitrate
            Color32::LIGHT_RED,    // NFR
            Color32::LIGHT_YELLOW, // RTT
            Color32::LIGHT_BLUE,   // volatility
            Color32::LIGHT_GRAY,   // fairness
        ];

        // Labels for display in tooltip
        let labels = vec!["Bitrate", "NFR", "RTT", "Volatility", "Fairness"];

        let mut all_values = Vec::new();
        for snap in &self.history_sarsa {
            for v in &snap.r_components {
                all_values.push(*v as f64);
            }
        }

        let mut data = statistics::Data::new(all_values);
        let lower = data.quantile(0.0) as f32;
        let upper = data.quantile(1.0) as f32;

        let hist = &self.history_sarsa;
        let colors_paint = colors.clone();
        let colors_tooltip = colors.clone();
        let labels_tooltip = labels.clone();

        self.draw_sarsa_graph(
            ui,
            available_width,
            "SARSA Reward Components",
            lower..=upper,
            move |painter, to_screen_trans| {
                let hist_len = hist.len();
                if hist_len < 2 {
                    return;
                }

                for comp_idx in 0..num_components {
                    let color = colors_paint[comp_idx % colors_paint.len()];
                    let mut points = Vec::with_capacity(hist_len);

                    for i in 0..hist_len {
                        let stats = &hist[i];
                        let val = stats.r_components[comp_idx];
                        points.push(to_screen_trans * pos2(i as f32, val));
                    }

                    if points.len() > 1 {
                        draw_lines(painter, points, color);
                    }
                }
            },
            move |ui, stats| {
                for (idx, val) in stats.r_components.iter().enumerate() {
                    let color = colors_tooltip[idx % colors_tooltip.len()];
                    ui.colored_label(color, format!("{}: {:+.4}", labels_tooltip[idx], val));
                }

                ui.separator();

                let total: f32 = stats.r_components.iter().sum();
                ui.label(format!("Total reward: {:.4}", total));
            },
        );
    }

    make_ap_custom_graph!(
        fn draw_ap_clients_rx_mcs_graph(self, ui, width),
        title = "RX MCS (all clients)",
        yrange = 0.0..=12.0,

        paint = |painter: &Painter, to_screen_trans, _i, _iface: &Interface, clients: &Vec<Client>, _client| {
            for client in clients {
                let color = series_color(&client.ip);

                let points: Vec<Pos2> = (0..self.history_ap.len())
                    .enumerate()
                    .filter_map(|(idx, _)| {
                        let snap = &self.history_ap[idx];
                        snap.interfaces.iter()
                            .flat_map(|iface| &iface.clients)
                            .find(|c| c.ip == client.ip)
                            .and_then(|c| c.rx.mcs.map(|mcs| to_screen_trans * pos2(idx as f32, mcs as f32)))
                    })
                    .collect();

                if points.len() >= 2 {
                    draw_lines(painter, points, color);
                }
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                tui.label(format!("Interface: {}", iface.interface));
                for c in &iface.clients {
                    let color = series_color(&c.ip);
                    tui.colored_label(color, format!("{}: RX MCS {}", c.ip, c.rx.mcs.map_or("N/A".to_string(), |v| v.to_string())));
                }
            }
        }
    );

    make_ap_custom_graph!(
        fn draw_ap_clients_tx_mcs_graph(self, ui, width),
        title = "TX MCS (all clients)",
        yrange = 0.0..=12.0,

        paint = |painter: &Painter, to_screen_trans, _i, _iface: &Interface, clients: &Vec<Client>, _client| {
            for client in clients {
                let color = series_color(&client.ip);

                let points: Vec<Pos2> = (0..self.history_ap.len())
                    .enumerate()
                    .filter_map(|(idx, _)| {
                        let snap = &self.history_ap[idx];
                        snap.interfaces.iter()
                            .flat_map(|iface| &iface.clients)
                            .find(|c| c.ip == client.ip)
                            .and_then(|c| c.tx.mcs.map(|mcs| to_screen_trans * pos2(idx as f32, mcs as f32)))
                    })
                    .collect();

                if points.len() >= 2 {
                    draw_lines(painter, points, color);
                }
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                tui.label(format!("Interface: {}", iface.interface));
                for c in &iface.clients {
                    let color = series_color(&c.ip);
                    tui.colored_label(color, format!("{}: TX MCS {}", c.ip, c.tx.mcs.map_or("N/A".to_string(), |v| v.to_string())));
                }
            }
        }
    );

    make_ap_custom_graph!(
        fn draw_ap_interface_channel_activity_graph(self, ui, width),
        title = "Activity (%)",
        yrange = 0.0..=100.0,

        paint = |painter: &Painter, to_screen_trans, _i, iface: &Interface, _clients, _client| {
            let hist = &self.history_ap;
            if hist.len() < 2 { return; }

            let mut busy_points = Vec::new();
            let mut rx_points   = Vec::new();
            let mut tx_points   = Vec::new();

            // Store previous counters
            let mut prev_busy   = None;
            let mut prev_rx     = None;
            let mut prev_tx     = None;
            let mut prev_active = None;

            for (idx, ap_snapshot) in hist.iter().enumerate() {
                if let Some(hist_iface) = ap_snapshot
                    .interfaces
                    .iter()
                    .find(|i| i.interface == iface.interface)
                {
                    let act = match hist_iface.ch_active_time_ms {
                        Some(v) if v > 0 => v,
                        _ => continue, // skip invalid/missing snapshot
                    };
                    let busy = match hist_iface.ch_busy_time_ms { Some(v) => v, None => continue };
                    let rx   = match hist_iface.ch_rx_time_ms   { Some(v) => v, None => continue };
                    let tx   = match hist_iface.ch_tx_time_ms   { Some(v) => v, None => continue };

                    if let (Some(p_busy), Some(p_active)) =
                        (prev_busy, prev_active)
                    {
                        let delta_active = act - p_active;
                        if delta_active > 0 {
                            let x = idx as f32;

                            let busy_p = (busy - p_busy)  as f32 * 100.0 / delta_active as f32;
                            busy_points.push(to_screen_trans * pos2(x, busy_p));

                            if self.bulk_ap_stats {
                                if let (Some(p_rx), Some(p_tx)) = (prev_rx, prev_tx) {
                                    rx_points.push(to_screen_trans * pos2(x, (rx - p_rx) as f32 * 100.0 / delta_active as f32));
                                    tx_points.push(to_screen_trans * pos2(x, (tx - p_tx) as f32 * 100.0 / delta_active as f32));
                                }
                            }
                        }
                    }

                    // Update previous counters
                    prev_busy   = Some(busy);
                    prev_rx     = Some(rx);
                    prev_tx     = Some(tx);
                    prev_active = Some(act);
                }
            }

            if busy_points.len() >= 2 { draw_lines(painter, busy_points, Color32::RED); }
            if self.bulk_ap_stats {
                if rx_points.len() >= 2   { draw_lines(painter, rx_points, Color32::GREEN); }
                if tx_points.len() >= 2   { draw_lines(painter, tx_points, Color32::BLUE); }
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                let idx = self.history_ap.iter().position(|s| std::ptr::eq(s, stats));
                if idx.is_none() || idx.unwrap() == 0 {
                    // No delta possible at index 0
                    return;
                }
                let i = idx.unwrap();

                let prev = &self.history_ap[i - 1];
                let curr = &self.history_ap[i];

                // Find matching interface in prev/curr snapshots
                let prev_iface = prev.interfaces.iter().find(|x| x.interface == iface.interface);
                let curr_iface = curr.interfaces.iter().find(|x| x.interface == iface.interface);

                if prev_iface.is_none() || curr_iface.is_none() {
                    return;
                }

                let p = prev_iface.unwrap();
                let c = curr_iface.unwrap();

                if let (Some(p_act), Some(c_act),
                        Some(p_busy), Some(c_busy)) =
                    (p.ch_active_time_ms, c.ch_active_time_ms,
                    p.ch_busy_time_ms,   c.ch_busy_time_ms)
                {
                    let delta_active = c_act - p_act;
                    if delta_active <= 0 { return; }

                    let busy_p = ((c_busy - p_busy) as f32 * 100.0 / delta_active as f32).clamp(0.0, 100.0);
                    tui.colored_label(Color32::RED, format!("Busy (channel): {:.1}%", busy_p));

                    if self.bulk_ap_stats {
                        if let (Some(p_rx), Some(c_rx)) = (p.ch_rx_time_ms, c.ch_rx_time_ms) {
                            let rx_p = ((c_rx - p_rx) as f32 * 100.0 / delta_active as f32)
                                .clamp(0.0, 100.0);
                            tui.colored_label(Color32::GREEN, format!("RX: {:.1}%", rx_p));
                        }

                        if let (Some(p_tx), Some(c_tx)) = (p.ch_tx_time_ms, c.ch_tx_time_ms) {
                            let tx_p = ((c_tx - p_tx) as f32 * 100.0 / delta_active as f32)
                                .clamp(0.0, 100.0);
                            tui.colored_label(Color32::BLUE, format!("TX: {:.1}%", tx_p));
                        }
                    }
                }
            }
        }
    );

    make_ap_custom_graph!(
        fn draw_ap_clients_airtime_graph(self, ui, width),
        title = "Client Airtime (%)",
        yrange = 0.0..=100.0,

        paint = |painter: &Painter, to_screen_trans, _i, _iface: &Interface, clients: &Vec<Client>, _client| {
            let hist = &self.history_ap;
            if hist.len() < 2 { return; }

            for client in clients {
                let color = series_color(&client.ip);
                let mut points = Vec::new();

                let mut prev_tx = None;
                let mut prev_rx = None;
                let mut prev_t  = None;

                for (idx, snap) in hist.iter().enumerate() {
                    // Lookup this client in this snapshot
                    let c = snap.interfaces
                        .iter()
                        .flat_map(|iface| &iface.clients)
                        .find(|c| c.ip == client.ip);

                    if let Some(c) = c {
                        let (tx_us, rx_us, now_ms) = match (c.tx.duration, c.rx.duration, c.current_time_ms) {
                            (Some(tx), Some(rx), Some(t)) => (tx, rx, t),
                            _ => continue, // skip this snapshot for this client
                        };

                        if let (Some(p_tx), Some(p_rx), Some(p_time)) = (prev_tx, prev_rx, prev_t) {

                            // detect reset / invalid delta
                            if now_ms > p_time && tx_us >= p_tx && rx_us >= p_rx {
                                let dt = (now_ms - p_time) * 1000; // ms → μs
                                if dt > 0 {
                                    let delta_air = (tx_us - p_tx + rx_us - p_rx) as f32;
                                    let airtime = (delta_air / dt as f32) * 100.0;

                                    points.push(to_screen_trans * pos2(idx as f32, airtime.clamp(0.0, 100.0)));
                                }
                            }
                        }

                        prev_tx = Some(tx_us);
                        prev_rx = Some(rx_us);
                        prev_t  = Some(now_ms);
                    }
                }

                if points.len() >= 2 {
                    draw_lines(painter, points, color);
                }
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                let idx = self.history_ap.iter().position(|s| std::ptr::eq(s, stats));
                if idx.is_none() || idx.unwrap() == 0 {
                    // No delta possible at index 0
                    return;
                }
                let i = idx.unwrap();

                let prev_snap = &self.history_ap[i - 1];
                let curr_snap = &self.history_ap[i];

                let prev_iface = prev_snap.interfaces.iter().find(|x| x.interface == iface.interface);
                let curr_iface = curr_snap.interfaces.iter().find(|x| x.interface == iface.interface);

                if prev_iface.is_none() || curr_iface.is_none() { return; }
                let p_if = prev_iface.unwrap();
                let c_if = curr_iface.unwrap();

                for curr in &c_if.clients {
                    let color = series_color(&curr.ip);

                    let prev = p_if.clients.iter().find(|cl| cl.ip == curr.ip);
                    if prev.is_none() {
                        tui.colored_label(color, format!("{}: (no delta yet)", curr.ip));
                        continue;
                    }

                    let prev = prev.unwrap();

                    if let (Some(p_tx), Some(p_rx), Some(p_t),
                            Some(c_tx), Some(c_rx), Some(c_t)) =
                        (prev.tx.duration, prev.rx.duration, prev.current_time_ms,
                            curr.tx.duration, curr.rx.duration, curr.current_time_ms)
                    {
                        if c_t <= p_t || c_tx < p_tx || c_rx < p_rx {
                            tui.colored_label(color, format!("{}: (reset)", curr.ip));
                            continue;
                        }

                        let dt = (c_t - p_t) * 1000;
                        if dt == 0 {
                            tui.colored_label(color, format!("{}: (no update)", curr.ip));
                            continue;
                        }

                        let delta_tx = c_tx - p_tx;
                        let delta_rx = c_rx - p_rx;
                        let airtime = ((delta_tx + delta_rx) as f32 / dt as f32) * 100.0;

                        tui.colored_label(
                            color,
                            format!(
                                "{}: {:.1}%  (ΔTX {}µs, ΔRX {}µs)",
                                curr.ip,
                                airtime.clamp(0.0, 100.0),
                                delta_tx,
                                delta_rx
                            )
                        );
                    }
                }
            }
        }
    );

    make_ap_series_graph!(
        fn draw_ap_clients_count_graph(self, ui, width),
        title = "Client Count",
        yrange = 0.0..=10.0,

        series = {
            "count" => |iface: &Interface, _client: Option<&Client>| iface.clients.len() as f32
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                let color = series_color("count");
                tui.colored_label(color, format!("Clients: {}", iface.clients.len()));

                for c in &iface.clients {
                    if self.bulk_ap_stats {
                        tui.label(format!(
                            "{}: {} dBm (vr? {})",
                            c.ip,
                            c.signal_dbm.map_or("N/A".to_string(), |v| v.to_string()),
                            c.is_vr.map_or("N/A".to_string(), |v| v.to_string())
                        ));
                    } else {
                        tui.label(format!("{} (vr? {})", c.ip, c.is_vr.map_or("N/A".to_string(), |v| v.to_string())));
                    }
                }
            }
        }
    );

    make_ap_series_graph!(
        fn draw_ap_client_snr_graph(self, ui, width),
        title = "Client SNR (dB)",
        yrange = 0.0..=60.0,

        series = {
            "snr" => |_iface: &Interface, client: Option<&Client>| {
                client.and_then(|c| c.snr_db.map(|v| v as f32)).unwrap_or(f32::NAN)
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, client_opt) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(_iface) = iface_opt {
                if let Some(client) = client_opt {
                    let color = series_color("snr");
                    tui.colored_label(
                        color,
                        format!(
                            "SNR: {} dB",
                            client.snr_db.map_or("N/A".to_string(), |v| v.to_string())
                        )
                    );
                    tui.label(format!(
                        "Signal: {} dBm",
                        client.signal_dbm.map_or("N/A".to_string(), |v| v.to_string())
                    ));
                    tui.label(format!(
                        "Noise: {} dBm",
                        client.noise_dbm.map_or("N/A".to_string(), |v| v.to_string())
                    ));
                }
            }
        }
    );

    make_ap_series_graph!(
        fn draw_ap_client_tx_rx_mcs_graph(self, ui, width),
        title = "Client TX/RX MCS",
        yrange = 0.0..=12.0,

        series = {
            "rx_mcs" => |_iface: &Interface, client: Option<&Client>| {
                client.and_then(|c| c.rx.mcs.map(|v| v as f32)).unwrap_or(f32::NAN)
            },
            "tx_mcs" => |_iface: &Interface, client: Option<&Client>| {
                if self.bulk_ap_stats {
                    client.and_then(|c| c.tx.mcs.map(|v| v as f32)).unwrap_or(f32::NAN)
                } else {
                    f32::NAN
                }
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, client_opt) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(_iface) = iface_opt {
                if let Some(client) = client_opt {
                    let color = series_color("rx_mcs");
                    tui.colored_label(color, format!(
                        "RX MCS: {}",
                        client.rx.mcs.map_or("N/A".to_string(), |v| v.to_string())
                    ));
                    tui.label(format!(
                        "RX bitrate: {} Mbps",
                        client.rx.bitrate_mbps.map_or("N/A".to_string(), |v| v.to_string())
                    ));
                    if self.bulk_ap_stats {
                        tui.colored_label(series_color("tx_mcs"), format!(
                            "TX MCS: {}",
                            client.tx.mcs.map_or("N/A".to_string(), |v| v.to_string())
                        ));
                        tui.label(format!(
                            "TX bitrate: {} Mbps",
                            client.tx.bitrate_mbps.map_or("N/A".to_string(), |v| v.to_string())
                        ));
                    }
                }
            }
        }
    );

    make_ap_series_graph!(
        fn draw_ap_interface_bytes_mbps_graph(self, ui, width),
        title = "Interface RX/TX Mbps",
        yrange = {
            let max_mbps = self.history_ap
                .iter()
                .flat_map(|ap| &ap.interfaces)
                .map(|iface| {
                    let rx = iface.rx_kbytes_s.unwrap_or(0.0) as f32 * 8.0 / 1000.0;
                    let tx = iface.tx_kbytes_s.unwrap_or(0.0) as f32 * 8.0 / 1000.0;
                    rx.max(tx)
                })
                .fold(0.0f32, |a, b| a.max(b));
            0.0..=(max_mbps * 1.1) // add 10% padding
        },

        series = {
            "rx_mbps" => |iface: &Interface, _| {
                iface.rx_kbytes_s.map(|v| v as f32 * 8.0 / 1000.0).unwrap_or(f32::NAN)
            },
            "tx_mbps" => |iface: &Interface, _| {
                iface.tx_kbytes_s.map(|v| v as f32 * 8.0 / 1000.0).unwrap_or(f32::NAN)
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                tui.colored_label(series_color("rx_mbps"), format!(
                    "RX: {} Mbps",
                    iface.rx_kbytes_s.map_or("N/A".to_string(), |v| format!("{:.2}", v as f32 * 8.0 / 1000.0))
                ));
                tui.colored_label(series_color("tx_mbps"), format!(
                    "TX: {} Mbps",
                    iface.tx_kbytes_s.map_or("N/A".to_string(), |v| format!("{:.2}", v as f32 * 8.0 / 1000.0))
                ));
            }
        }
    );

    make_ap_series_graph!(
        fn draw_ap_interface_quality_graph(self, ui, width),
        title = "Interface Link Quality",
        yrange = 0.0..=100.0,

        series = {
            "link_quality" => |iface: &Interface, _| {
                iface.link_quality.as_deref()
                    .and_then(|s| s.split_once('/'))
                    .and_then(|(num, den)| {
                        let a = num.parse::<f32>().ok()?;
                        let b = den.parse::<f32>().ok()?;
                        Some(100.0 * a / b)
                    })
                    .unwrap_or(f32::NAN) // skip plotting if parsing fails
            }
        },

        tooltip = |tui, stats| {
            let (iface_opt, _) = find_client_interface(stats, self.client_ip.unwrap());
            if let Some(iface) = iface_opt {
                let link_quality_percent = iface.link_quality
                    .as_deref()
                    .and_then(|s| s.split_once('/'))
                    .and_then(|(num, den)| {
                        let a = num.parse::<f32>().ok()?;
                        let b = den.parse::<f32>().ok()?;
                        Some(100.0 * a / b)
                    });

                let link_quality_display = iface.link_quality.as_deref().unwrap_or("N/A");

                tui.colored_label(
                    series_color("link_quality"),
                    format!(
                        "Link Quality: {} ({})",
                        link_quality_display,
                        link_quality_percent.map_or("N/A".to_string(), |v| format!("{:.1}%", v))
                    )
                );

                tui.label(format!(
                    "Utilization: {}",
                    iface.if_util.map_or("N/A".to_string(), |v| format!("{:.1}", v as f32))
                ));
                tui.label(format!(
                    "Signal: {} dBm, Noise: {} dBm",
                    iface.signal_dbm.map_or("N/A".to_string(), |v| v.to_string()),
                    iface.noise_dbm.map_or("N/A".to_string(), |v| v.to_string())
                ));
            }
        }
    );

    fn draw_network_graph(
        &self,
        ui: &mut Ui,
        available_width: f32,
        title: &str,
        data_range: RangeInclusive<f32>,
        graph_content: impl FnOnce(&Painter, RectTransform),
        tooltip_content: impl FnOnce(&mut Ui, &GraphNetworkStatistics),
    ) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(20.0));

        let canvas_response = Frame::canvas(ui.style()).show(ui, |ui| {
            ui.ctx().request_repaint();
            let size = available_width * vec2(1.0, 0.2);

            let (_id, canvas_rect) = ui.allocate_space(size);

            let max = *data_range.end();
            let min = *data_range.start();
            let data_rect = Rect::from_x_y_ranges(0.0..=GRAPH_HISTORY_SIZE as f32, max..=min);
            let to_screen = RectTransform::from_to(data_rect, canvas_rect);

            let painter = ui.painter().with_clip_rect(canvas_rect);

            graph_content(&painter, to_screen);

            ui.painter().text(
                to_screen * pos2(0.0, min),
                Align2::LEFT_BOTTOM,
                format!("{:.0}", min),
                FontId::monospace(20.0),
                Color32::GRAY,
            );
            ui.painter().text(
                to_screen * pos2(0.0, max),
                Align2::LEFT_TOP,
                format!("{:.0}", max),
                FontId::monospace(20.0),
                Color32::GRAY,
            );

            data_rect
        });

        if let Some(pos) = canvas_response.response.hover_pos() {
            let graph_pos =
                RectTransform::from_to(canvas_response.response.rect, canvas_response.inner) * pos;
            let history_index = (graph_pos.x as usize).clamp(0, GRAPH_HISTORY_SIZE - 1);

            popup::show_tooltip(ui.ctx(), Id::new("popup"), |ui| {
                tooltip_content(ui, self.history_network.get(history_index).unwrap())
            });
        }
    }

    fn draw_latency_graph(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history
                .iter()
                .map(|stats| stats.total_pipeline_latency_s as f64)
                .collect::<Vec<_>>(),
        );

        self.draw_graph(
            ui,
            available_width,
            "Latency (ms)",
            0.0..=(data.quantile(UPPER_QUANTILE)) as f32 * 1000.0,
            |painter, to_screen_trans| {
                for i in 0..GRAPH_HISTORY_SIZE {
                    let stats = self.history.get(i).unwrap();
                    let mut offset = 0.0;
                    for (value, color) in &[
                        (stats.game_time_s, graph_colors::RENDER_VARIANT),
                        (stats.server_compositor_s, graph_colors::RENDER),
                        (stats.encoder_s, graph_colors::TRANSCODE),
                        (stats.network_s, graph_colors::NETWORK),
                        (stats.decoder_s, graph_colors::TRANSCODE),
                        (stats.decoder_queue_s, graph_colors::IDLE),
                        (stats.client_compositor_s, graph_colors::RENDER),
                        (stats.vsync_queue_s, graph_colors::IDLE),
                    ] {
                        painter.rect_filled(
                            Rect {
                                min: to_screen_trans * pos2(i as f32, offset + value * 1000.0),
                                max: to_screen_trans * pos2(i as f32 + 2.0, offset),
                            },
                            Rounding::ZERO,
                            *color,
                        );
                        offset += value * 1000.0;
                    }
                }
            },
            |ui, stats| {
                use graph_colors::*;

                fn label(ui: &mut Ui, text: &str, value_s: f32, color: Color32) {
                    ui.colored_label(color, &format!("{text}: {:.2} ms", value_s * 1000.0));
                }
                label(
                    ui,
                    "Total latency",
                    stats.total_pipeline_latency_s,
                    theme::FG,
                );
                label(ui, "Client VSync", stats.vsync_queue_s, IDLE);
                label(ui, "Client compositor", stats.client_compositor_s, RENDER);
                label(ui, "Decoder queue", stats.decoder_queue_s, IDLE);
                label(ui, "Decode", stats.decoder_s, TRANSCODE);
                label(ui, "Network", stats.network_s, NETWORK);
                label(ui, "Encode", stats.encoder_s, TRANSCODE);
                label(ui, "Streamer compositor", stats.server_compositor_s, RENDER);
                label(ui, "Game render", stats.game_time_s, RENDER_VARIANT);
            },
        );
    }

    fn draw_fps_graph(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.client_fps)
                .chain(self.history_network.iter().map(|stats| stats.server_fps))
                .map(|v| v as f64)
                .collect::<Vec<_>>(),
        );
        let upper_quantile = data.quantile(UPPER_QUANTILE);
        let lower_quantile = data.quantile(1.0 - UPPER_QUANTILE);

        let max = upper_quantile + (upper_quantile - lower_quantile);
        let min = 0.0;

        self.draw_network_graph(
            ui,
            available_width,
            "Framerate",
            min as f32..=max as f32,
            |painter, to_screen_trans| {
                let (server_fps_points, client_fps_points) = (0..GRAPH_HISTORY_SIZE)
                    .map(|i| {
                        (
                            to_screen_trans * pos2(i as f32, self.history_network[i].server_fps),
                            to_screen_trans * pos2(i as f32, self.history_network[i].client_fps),
                        )
                    })
                    .unzip();

                draw_lines(painter, server_fps_points, graph_colors::SERVER_FPS);
                draw_lines(painter, client_fps_points, graph_colors::CLIENT_FPS);
            },
            |ui, stats| {
                ui.colored_label(
                    graph_colors::SERVER_FPS,
                    format!("Streamer FPS: {:.2}", stats.server_fps),
                );
                ui.colored_label(
                    graph_colors::CLIENT_FPS,
                    format!("Client FPS: {:.2}", stats.client_fps),
                );
            },
        );
    }

    fn draw_jitter(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.ow_delay_ms as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_network_graph(
            ui,
            available_width,
            "Shards Jitter Graph",
            -5.0..=(data.quantile(UPPER_QUANTILE) * 5.0) as f32,
            |painter, to_screen_trans| {
                let mut interarrival_jitter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut ow_delay = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut filtered_ow_delay = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                // let mut threshold_gcc = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];

                    let value_fowd = pointer_graphstatistics.filtered_ow_delay_ms;
                    filtered_ow_delay.push(to_screen_trans * pos2(i as f32, value_fowd));

                    let value_jitt = pointer_graphstatistics.interarrival_jitter_ms;
                    interarrival_jitter.push(to_screen_trans * pos2(i as f32, value_jitt));

                    let value_owd = pointer_graphstatistics.ow_delay_ms;
                    ow_delay.push(to_screen_trans * pos2(i as f32, value_owd));

                    // let value_thr = pointer_graphstatistics.threshold_gcc;
                    // threshold_gcc.push(to_screen_trans * pos2(i as f32, value_thr));
                }
                draw_lines(painter, filtered_ow_delay, Color32::LIGHT_YELLOW);
                draw_lines(painter, interarrival_jitter, Color32::RED);
                draw_lines(painter, ow_delay, Color32::BLUE);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.7} ms", value));
                    }
                }
                maybe_label(
                    ui,
                    "Filtered OW Delay",
                    Some(stats.filtered_ow_delay_ms),
                    Color32::LIGHT_YELLOW,
                );
                maybe_label(
                    ui,
                    "Shard Interarrival Jitter",
                    Some(stats.interarrival_jitter_ms),
                    Color32::RED,
                );
                maybe_label(ui, "OW Delay", Some(stats.ow_delay_ms), Color32::BLUE);

                // maybe_label(
                //     ui,
                //     "Threshold from GCC",
                //     Some(stats.threshold_gcc),
                //     Color32::GOLD,
                // );
            },
        )
    }

    fn draw_frameloss(&self, ui: &mut Ui, available_width: f32) {
        self.draw_network_graph(
            ui,
            available_width,
            "Frames Skipped, Shards Lost and Shards Duplicated Graph",
            -20.0..=20.0 as f32,
            |painter, to_screen_trans| {
                let mut frameskipped = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut shardloss = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut dup_shards = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];

                    let val_fs = pointer_graphstatistics.frames_skipped;
                    frameskipped.push(to_screen_trans * pos2(i as f32, val_fs as f32));

                    let val_sl = pointer_graphstatistics.shards_lost;
                    shardloss.push(to_screen_trans * pos2(i as f32, val_sl as f32));

                    let val_dups = pointer_graphstatistics.shards_duplicated;
                    dup_shards.push(to_screen_trans * pos2(i as f32, val_dups as f32));
                }

                draw_lines(painter, frameskipped, Color32::LIGHT_BLUE);
                draw_lines(painter, shardloss, Color32::LIGHT_RED);
                draw_lines(painter, dup_shards, Color32::DARK_GREEN);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.0} ", value));
                    }
                }
                let graphstats = stats;
                maybe_label(
                    ui,
                    "Frames Skipped",
                    Some(graphstats.frames_skipped as f32),
                    Color32::LIGHT_BLUE,
                );
                maybe_label(
                    ui,
                    "Shards Lost",
                    Some(graphstats.shards_lost as f32),
                    Color32::LIGHT_RED,
                );
                maybe_label(
                    ui,
                    "Shards Duplicated",
                    Some(graphstats.shards_duplicated as f32),
                    Color32::DARK_GREEN,
                );
            },
        )
    }

    fn draw_frame_span_interarrival(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.frame_interarrival_ms as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_network_graph(
            ui,
            available_width,
            "Frame Span and Frame Interarrival Graph",
            0.0..=(data.quantile(UPPER_QUANTILE) * 2.0) as f32,
            |painter, to_screen_trans| {
                let mut frame_span = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut frame_interarrival = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut frame_jitter = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];

                    let fs = pointer_graphstatistics.frame_span_ms;
                    frame_span.push(to_screen_trans * pos2(i as f32, fs as f32));

                    let fi = pointer_graphstatistics.frame_interarrival_ms;
                    frame_interarrival.push(to_screen_trans * pos2(i as f32, fi as f32));

                    let val_std = pointer_graphstatistics.frame_jitter_ms;
                    frame_jitter.push(to_screen_trans * pos2(i as f32, val_std));
                }
                draw_lines(painter, frame_interarrival, Color32::LIGHT_RED);
                draw_lines(painter, frame_span, Color32::LIGHT_BLUE);
                draw_lines(painter, frame_jitter, Color32::LIGHT_YELLOW);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.6} ms", value));
                    }
                }
                let graphstats = stats;

                maybe_label(
                    ui,
                    "Frame span",
                    Some(graphstats.frame_span_ms as f32),
                    Color32::LIGHT_BLUE,
                );
                maybe_label(
                    ui,
                    "Frame Interarrival",
                    Some(graphstats.frame_interarrival_ms as f32),
                    Color32::LIGHT_RED,
                );
                maybe_label(
                    ui,
                    "Frame interarrival std (frame jitter).",
                    Some(stats.frame_jitter_ms),
                    Color32::LIGHT_YELLOW,
                );
            },
        )
    }

    fn draw_throughput_graphs(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.interval_avg_plot_throughput as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_network_graph(
            ui,
            available_width,
            "Video Network Throughput",
            0.0..=(data.quantile(UPPER_QUANTILE) * 2.0) as f32 / 1e6,
            |painter, to_screen_trans| {
                let mut network_throughput_bps: Vec<Pos2> = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                let mut requested = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];
                    let nom_br = &self.history_network[i].nominal_bitrate;

                    let value_nw = pointer_graphstatistics.interval_avg_plot_throughput;
                    network_throughput_bps.push(to_screen_trans * pos2(i as f32, value_nw / 1e6));

                    requested.push(to_screen_trans * pos2(i as f32, nom_br.requested_bps / 1e6));
                }
                draw_lines(painter, network_throughput_bps, Color32::BLUE);
                draw_lines(painter, requested, theme::OK_GREEN);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.4} Mbps", value / 1e6));
                    }
                }
                let graphstats = stats;
                let n = &stats.nominal_bitrate;

                maybe_label(
                    ui,
                    "Network Throughput",
                    Some(graphstats.interval_avg_plot_throughput),
                    Color32::BLUE,
                );
                maybe_label(
                    ui,
                    "Requested Bitrate",
                    Some(n.requested_bps),
                    theme::OK_GREEN,
                );
            },
        )
    }

    fn draw_bitrate_graph(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history
                .iter()
                .map(|stats| stats.actual_bitrate_bps as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_graph(
            ui,
            available_width,
            "Bitrate (ALVR's computation)",
            0.0..=(data.quantile(UPPER_QUANTILE) * 2.0) as f32 / 1e6,
            |painter, to_screen_trans| {
                let mut scaled_calculated = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut decoder_latency_limiter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut network_latency_limiter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut encoder_latency_limiter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut manual_max = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut manual_min = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut requested = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut actual = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let nom_br = &self.history[i].nominal_bitrate;

                    if let Some(value) = nom_br.scaled_calculated_bps {
                        scaled_calculated.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.decoder_latency_limiter_bps {
                        decoder_latency_limiter.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.network_latency_limiter_bps {
                        network_latency_limiter.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.encoder_latency_limiter_bps {
                        encoder_latency_limiter.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.manual_max_bps {
                        manual_max.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.manual_min_bps {
                        manual_min.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }

                    requested.push(to_screen_trans * pos2(i as f32, nom_br.requested_bps / 1e6));
                    actual.push(
                        to_screen_trans * pos2(i as f32, self.history[i].actual_bitrate_bps / 1e6),
                    );
                }

                draw_lines(painter, scaled_calculated, Color32::GRAY);
                draw_lines(painter, encoder_latency_limiter, graph_colors::TRANSCODE);
                draw_lines(painter, network_latency_limiter, graph_colors::NETWORK);
                draw_lines(painter, decoder_latency_limiter, graph_colors::TRANSCODE);
                draw_lines(painter, manual_max, graph_colors::RENDER);
                draw_lines(painter, manual_min, graph_colors::RENDER);
                draw_lines(painter, requested, theme::OK_GREEN);
                draw_lines(painter, actual, theme::FG);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.2} Mbps", value / 1e6));
                    }
                }

                let n = &stats.nominal_bitrate;

                maybe_label(
                    ui,
                    "Initial calculated",
                    n.scaled_calculated_bps,
                    Color32::GRAY,
                );
                maybe_label(
                    ui,
                    "Encoder latency limiter",
                    n.encoder_latency_limiter_bps,
                    graph_colors::TRANSCODE,
                );
                maybe_label(
                    ui,
                    "Network latency limiter",
                    n.network_latency_limiter_bps,
                    graph_colors::NETWORK,
                );
                maybe_label(
                    ui,
                    "Decoder latency limiter",
                    n.decoder_latency_limiter_bps,
                    graph_colors::TRANSCODE,
                );
                maybe_label(ui, "Manual max", n.manual_max_bps, graph_colors::RENDER);
                maybe_label(ui, "Manual min", n.manual_min_bps, graph_colors::RENDER);
                maybe_label(ui, "Requested", Some(n.requested_bps), theme::OK_GREEN);
                maybe_label(
                    ui,
                    "Actual recorded",
                    Some(stats.actual_bitrate_bps),
                    theme::FG,
                );
            },
        )
    }

    fn draw_statistics_overview(&self, ui: &mut Ui, statistics: &StatisticsSummary) {
        ui.add_space(10.0);

        ui.columns(2, |ui| {
            ui[0].label("Total packets:");
            ui[1].label(&format!(
                "{} packets ({} packets/s)",
                statistics.video_packets_total, statistics.video_packets_per_sec
            ));

            ui[0].label("Total sent:");
            ui[1].label(&format!("{} MB", statistics.video_mbytes_total));

            ui[0].label("Bitrate:");
            ui[1].label(&format!("{:.1} Mbps", statistics.video_mbits_per_sec));

            ui[0].label("Throughput:");
            ui[1].label(&format!(
                "{:.1} Mbps",
                statistics.video_throughput_mbits_per_sec
            ));

            ui[0].label("Game delay:");
            ui[1].label(&format!("{:.2} ms", statistics.game_delay_average_ms));

            ui[0].label("Server compositor delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.server_compositor_delay_average_ms
            ));

            ui[0].label("Encoder delay:");
            ui[1].label(&format!("{:.2} ms", statistics.encode_delay_average_ms));

            ui[0].label("Network delay:");
            ui[1].label(&format!("{:.2} ms", statistics.network_delay_average_ms));

            ui[0].label("Decoder delay:");
            ui[1].label(&format!("{:.2} ms", statistics.decode_delay_average_ms));

            ui[0].label("Decoder queue delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.decoder_queue_delay_average_ms
            ));

            ui[0].label("Client compositor delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.client_compositor_average_ms
            ));

            ui[0].label("Vsync delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.vsync_queue_delay_average_ms
            ));

            ui[0].label("Total latency:");
            ui[1].label(&format!(
                "{:.0} ms",
                statistics.total_pipeline_latency_average_ms
            ));

            ui[0].label("Frame jitter:");
            ui[1].label(&format!("{:.0} ms", statistics.frame_jitter_ms));

            ui[0].label("Total packets dropped:");
            ui[1].label(&format!(
                "{} packets ({} packets/s)",
                statistics.packets_dropped_total, statistics.packets_dropped_per_sec
            ));

            ui[0].label("Total packets skipped:");
            ui[1].label(&format!(
                "{} packets ({} packets/s)",
                statistics.packets_skipped_total, statistics.packets_skipped_per_sec
            ));

            ui[0].label("Shard loss:");
            ui[1].label(&format!("{} %", statistics.shard_loss_rate * 100.));

            ui[0].label("Client FPS:");
            ui[1].label(&format!("{} FPS", statistics.client_fps));

            ui[0].label("Streamer FPS:");
            ui[1].label(&format!("{} FPS", statistics.server_fps));

            ui[0].label("Headset battery");
            ui[1].label(&format!(
                "{}% ({})",
                statistics.battery_hmd,
                if statistics.hmd_plugged {
                    "plugged"
                } else {
                    "unplugged"
                }
            ));
        });
    }
}
