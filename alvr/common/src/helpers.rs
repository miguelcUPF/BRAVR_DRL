/// Generates a logarithmic bitrate ladder with unique integer steps.
pub fn generate_log_ladder(min_mbps: f32, max_mbps: f32, steps: usize) -> Vec<f32> {
    if steps < 2 {
        return vec![min_mbps];
    }

    let mut ladder = Vec::with_capacity(steps);

    // We work in log space to ensure geometric progression
    let log_min = min_mbps.ln();
    let log_max = max_mbps.ln();
    let step_size = (log_max - log_min) / ((steps - 1) as f32);

    for i in 0..steps {
        // Calculate raw value: e^(min + i*step)
        let val = (log_min + (i as f32 * step_size)).exp();

        // Round to nearest integer to avoid messy floats like 4.00002
        ladder.push(val.round());
    }

    // 1. Remove duplicates caused by rounding at the low end
    // (e.g., 2.1 and 2.4 both rounding to 2.0)
    ladder.dedup();

    // 2. Ensure we didn't lose the exact bounds due to rounding/dedup
    // Force the first element to be min
    if let Some(first) = ladder.first_mut() {
        *first = min_mbps.round();
    }
    // Force the last element to be max
    if let Some(last) = ladder.last_mut() {
        *last = max_mbps.round();
    }

    // 3. Sort
    ladder.sort_by(|a, b| a.partial_cmp(b).unwrap());

    ladder
}
