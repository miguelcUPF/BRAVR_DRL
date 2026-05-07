#!/bin/bash

VR_MAC_LIST="/usr/bin/vr_macs.txt" # one mac per line

# Load VR MACs into memory
declare -A VR_MACS
if [ -f "$VR_MAC_LIST" ]; then
    while read -r mac; do
        mac_clean=$(echo "$mac" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
        [ -n "$mac_clean" ] && VR_MACS["$mac_clean"]=1
    done < "$VR_MAC_LIST"
fi

collect_network_stats() {
	# Initialize JSON output
	json_output="{\"interfaces\":["

	# Iterate over each wireless interface
	for interface in $(iwinfo |grep "ESSID:" | cut -d' ' -f1); do
		# Get interface details
		details=$(iwinfo "$interface" info)
		
		# Parse interface details
		essid=$(echo "$details" | grep "ESSID:" | awk '{print $3}' | tr -d '" ')
		mac=$(echo "$details" | grep "Access Point:" | awk '{print $3}' | tr -d ' ')
		mode=$(echo "$details" | grep "Mode:" | awk '{print $2}' | tr -d ' ')
		channel=$(echo "$details" | grep "Channel:" | awk '{print $4}' | tr -d ' ')
		channel_ghz=$(echo "$details" | grep "Channel:" | awk '{print $5}' | tr -d '( ')
		ht_mode=$(echo "$details" | grep "HT Mode:" | awk '{print $9}' | tr -d ' ')
		tx_power_dbm=$(echo "$details" | grep "Tx-Power:" | awk '{print $2}' | tr -d ' ')
		link_quality=$(echo "$details" | grep "Link Quality:" | awk '{print $6}' | tr -d ' ')
		signal_dbm=$(echo "$details" | grep "Signal:" | awk '{print $2}' | tr -d ' ')
		if [ "$signal_dbm" = "unknown" ]; then
			noise_dbm=$(echo "$details" | grep "Noise:" | awk '{print $4}' | tr -d ' ')
		else
			noise_dbm=$(echo "$details" | grep "Noise:" | awk '{print $5}' | tr -d ' ')

		fi
		bitrate_mbps=$(echo "$details" | grep "Bit Rate:" | awk '{print $3}' | tr -d ' ')

		# Get network rates for the last second
		sar_details=$(sar -n DEV 1 1 | grep $interface | head -n 1)

		rx_pck_s=$(echo "$sar_details" | awk '{print $3}' | tr -d ' ')
		tx_pck_s=$(echo "$sar_details" | awk '{print $4}' | tr -d ' ')
		rx_kbytes_s=$(echo "$sar_details" | awk '{print $5}' | tr -d ' ')
		tx_kbytes_s=$(echo "$sar_details" | awk '{print $6}' | tr -d ' ')
		rx_cmp_s=$(echo "$sar_details" | awk '{print $7}' | tr -d ' ')
		tx_cmp_s=$(echo "$sar_details" | awk '{print $8}' | tr -d ' ')
		rx_mcst_s=$(echo "$sar_details" | awk '{print $9}' | tr -d ' ')
		if_util=$(echo "$sar_details" | awk '{print $10}' | tr -d ' ')

		# Get interface survey data
		while read -r line; do
			if [[ -z "$line" ]]; then
				continue
			fi
			if echo "$line" | grep -Eq 'active'; then
				ch_active_time_ms=$(echo "$line" | awk '{print $(NF-1)}' | tr -d ' ')
			elif echo "$line" | grep -Eq 'busy'; then
				ch_busy_time_ms=$(echo "$line" | awk '{print $(NF-1)}' | tr -d ' ')
			elif echo "$line" | grep -Eq 'receive' && ! echo "$line" | grep -Eq 'BSS'; then
				ch_rx_time_ms=$(echo "$line" | awk '{print $(NF-1)}' | tr -d ' ')
			elif echo "$line" | grep -Eq 'BSS receive'; then
				ch_bss_rx_time_ms=$(echo "$line" | awk '{print $(NF-1)}' | tr -d ' ')
			elif echo "$line" | grep -Eq 'transmit'; then
				ch_tx_time_ms=$(echo "$line" | awk '{print $(NF-1)}' | tr -d ' ')
			fi
		done < <(iw dev $interface survey dump | grep -A 6 'frequency.*\[in use\]')

		# Initialize clients JSON
		clients_json="["

		while read -r line; do
			if [[ -z "$line" ]]; then
				continue
			fi
			
			# Read MAC address line
			if echo "$line" | grep -Eq 'SNR'; then
				mac_c=$(echo "$line" | awk '{print $1}' | tr -d ' ')

				# Skip lines where MAC address is "No"
				if [ "$mac_c" = "No" ]; then
					continue
				fi

				if [ "${VR_MACS[$mac_c]}" = "1" ]; then
                    is_vr=true
                else
                    is_vr=false
                fi

				signal_c_dbm=$(echo "$line" | awk '{print $2}' | tr -d ' ')
				noise_c_dbm=$(echo "$line" | awk '{print $5}' | tr -d ' ')
				snr_c_db=$(echo "$line" | awk '{print $8}' | tr -d ' )')
				last_comm_ms=$(echo "$line" | awk '{print $9}' | tr -d ' ')

				# Get client IP
				ip_c=$(ip neigh | grep -i "$mac_c" |  awk '$6 == "REACHABLE" || $6 == "STALE" {print $1}' | head -n 1)
				if [ -z "$ip_c" ]; then
					ip_c="N/A"
				fi

				# Get hostname
				if [ "$ip_c" != "N/A" ]; then
					hostname_c=$(nslookup "$ip_c" | grep "name" | awk '{print $4}' | tr -d ' ')
					if [ -z "$hostname_c" ]; then
						hostname_c="N/A"
					fi
				else
					hostname_c="N/A"
				fi
			fi
			
			# Read RX details
			if echo "$line" | grep -Eq 'RX'; then		
				rx_details=$(echo "$line" | grep "RX:")
				bitrate_mbps_rx=$(echo "$rx_details" | awk '{print $2}' | tr -d ' ')
				if echo "$rx_details" | grep -Eq 'MCS'; then
					mcs_rx=$(echo "$rx_details" | awk '{print $5}' | tr -d ' ,')
				else
					mcs_rx="N/A"
				fi
				if echo "$rx_details" | grep -Eq 'MHz'; then
					bandwidth_rx_mhz=$(echo "$rx_details" | awk '{print $6}' | tr -d ' MHz,')
				else
					bandwidth_rx_mhz="N/A"
				fi
				if echo "$rx_details" | grep -Eq 'NSS'; then
					ss_rx=$(echo "$rx_details" | awk '{print $8}' | sed 's/[^0-9]//g')
				else
					ss_rx="N/A"
				fi
			fi
			# Read TX details
			if echo "$line" | grep -Eq 'TX'; then		
				tx_details=$(echo "$line" | grep "TX:")
				bitrate_mbps_tx=$(echo "$tx_details" | awk '{print $2}' | tr -d ' ')
				if echo "$tx_details" | grep -Eq 'MCS'; then
					mcs_tx=$(echo "$tx_details" | awk '{print $5}' | tr -d ' ,')
				else
					mcs_tx="N/A"
				fi
				if echo "$tx_details" | grep -Eq 'MHz'; then
					bandwidth_tx_mhz=$(echo "$tx_details" | awk '{print $6}' | tr -d ' MHz,')
				else
					bandwidth_tx_mhz="N/A"
				fi
				if echo "$tx_details" | grep -Eq 'NSS'; then
					ss_tx=$(echo "$tx_details" | awk '{print $8}' | sed 's/[^0-9]//g')

				else
					ss_tx="N/A"
				fi
			fi

			if echo "$line" | grep -Eq 'expected'; then		
				thr_exp_mbps=$(echo "$line" | grep "expected" | awk '{print $3}' | tr -d ' ')

				# Read also other details from iw
				iw_details=$(iw dev "$interface" station dump | grep -A 32  -i "$mac_c")
				
				bytes_rx=$(echo "$iw_details" | grep "rx bytes" | awk '{print $(NF)}' | tr -d ' ')
				pkts_rx=$(echo "$iw_details" | grep "rx packets" | awk '{print $(NF)}' | tr -d ' ')

				bytes_tx=$(echo "$iw_details" | grep "tx bytes" | awk '{print $(NF)}' | tr -d ' ')
				pkts_tx=$(echo "$iw_details" | grep "tx packets" | awk '{print $(NF)}' | tr -d ' ')
				retries_tx=$(echo "$iw_details" | grep "tx retries" | awk '{print $(NF)}' | tr -d ' ')
				failed_tx=$(echo "$iw_details" | grep "tx failed" | awk '{print $(NF)}' | tr -d ' ')

				duration_tx=$(echo "$iw_details" | grep  "tx duration" | awk '{print $3}' | tr -d 'us')
				duration_rx=$(echo "$iw_details" | grep  "rx duration" | awk '{print $3}' | tr -d 'us')

				current_time_ms=$(echo "$iw_details" | grep "current time" | awk '{print $(NF-1)}' | tr -d ' ')

				# Append client details to JSON
				c_json=$(cat <<EOF
				{	"ip": "$ip_c",
					"mac": "$mac_c",
					"hostname": "$hostname_c",
					"signal_dbm": "$signal_c_dbm",
					"noise_dbm": "$noise_c_dbm",
					"snr_db": "$snr_c_db",
					"is_vr": $is_vr,
					"last_comm_ms": "$last_comm_ms",
					"current_time_ms": "$current_time_ms",
					"rx": {
						"bitrate_mbps": "$bitrate_mbps_rx",
						"mcs": "$mcs_rx",
						"bandwidth_mhz": "$bandwidth_rx_mhz",
						"ss": "$ss_rx",
						"packets": "$pkts_rx",
						"bytes": "$bytes_rx",
						"duration": "$duration_rx"
					},
					"tx": {
						"bitrate_mbps": "$bitrate_mbps_tx",
						"mcs": "$mcs_tx",
						"bandwidth_mhz": "$bandwidth_tx_mhz",
						"ss": "$ss_tx",
						"packets": "$pkts_tx",
						"bytes": "$bytes_tx",
						"retries": "$retries_tx",
						"failed": "$failed_tx",
						"duration": "$duration_tx"
					},
					"expected_throughput_mbps": "$thr_exp_mbps"
				},
EOF
				)

				clients_json="$clients_json$c_json"
			fi
		
		done < <(iwinfo "$interface" assoclist)

		# Remove the trailing comma and close the  array
		clients_json=$(echo "$clients_json" | sed '$ s/,$//')
		clients_json="$clients_json]"
		
		# Append interface details to JSON
		i_json=$(cat <<EOF
		{	"interface": "$interface",
			"mac": "$mac",
			"essid": "$essid",
			"mode": "$mode",
			"channel": "$channel",
			"channel_ghz": "$channel_ghz",
			"ht_mode": "$ht_mode",
			"tx_power_dbm": "$tx_power_dbm",
			"link_quality": "$link_quality",
			"signal_dbm": "$signal_dbm",
			"noise_dbm": "$noise_dbm",
			"bitrate_mbps": "$bitrate_mbps",
			"rx_pck_s": "$rx_pck_s",
			"tx_pck_s": "$tx_pck_s",
			"rx_kbytes_s": "$rx_kbytes_s",
			"tx_kbytes_s": "$tx_kbytes_s",
			"rx_cmp_s": "$rx_cmp_s",
			"tx_cmp_s": "$tx_cmp_s",
			"rx_mcst_s": "$rx_mcst_s",
			"if_util": "$if_util",
			"ch_active_time_ms": "$ch_active_time_ms",
			"ch_busy_time_ms": "$ch_busy_time_ms",
			"ch_rx_time_ms": "$ch_rx_time_ms",
			"ch_bss_rx_time_ms": "$ch_bss_rx_time_ms",
			"ch_tx_time_ms": "$ch_tx_time_ms",
			"clients": $clients_json
		},
EOF
		)

		json_output="$json_output$i_json"

	done

	# Remove the trailing comma and close the JSON
	json_output=$(echo "$json_output" | sed '$s/,$//')  # Remove trailing comma
	json_output="$json_output]}"

	# Output the final JSON
	echo "$json_output" | jq .
}

cleanup() {
    # Kill the socat background process
    if [ -n "$REQUEST_PID" ]; then
        kill "$REQUEST_PID"
    fi

    # Kill processes associated with the port
    PID=$(netstat -tulnp | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$PID" ]; then
        kill -9 "$PID"
    fi

    # Kill the script itself
    pkill -f "$SCRIPT_PATH"

    # Exit the script
    exit 1
}

trap cleanup INT

SCRIPT_PATH=$(readlink -f "$0")

# Default PORT and DURATION VALUES
DEFAULT_PORT=8080
DEFAULT_DURATION=0 #infinite

# Read PORT and DURATION from command-line arguments
PORT=${1:-$DEFAULT_PORT}
DURATION=${2:-$DEFAULT_DURATION}

# Check if the DURATION is valid
if ! [ "$DURATION" -ge 0 ] 2>/dev/null; then
    echo "Invalid duration. It should be a non-negative integer."
    exit 1
fi

if [ "$DURATION" -ne 0 ]; then
    start_time=$(date +%s)
    end_time=$((start_time + DURATION))
fi

echo "Listening on port $PORT..."
while true; do 
	echo -e "HTTP/1.1 200 OK\nContent-Type: text/plain\n\n$(collect_network_stats)" | socat - TCP4-LISTEN:$PORT,reuseaddr
done &
REQUEST_PID=$!


while true; do 
    current_time=$(date +%s)

    if [ "$DURATION" -ne 0 ] && [ "$current_time" -ge "$end_time" ]; then
        echo "Duration $DURATION seconds has elapsed. Exiting..."
		cleanup
    fi 
done