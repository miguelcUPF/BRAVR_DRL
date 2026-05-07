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
	for interface in $(iwinfo | grep "ESSID:" | awk '{print $1}'); do
        ch_busy_time_ms=$(iw dev "$interface" survey dump | grep -A6 '\[in use\]' | grep 'busy time' | awk '{print $(NF-1)}')
        ch_active_time_ms=$(iw dev "$interface" survey dump | grep -A6 '\[in use\]' | grep 'active time' | awk '{print $(NF-1)}')

		# Initialize clients JSON
		clients_json="["

		while read -r line; do
            if echo "$line" | grep -Eq 'SNR'; then
                mac_c=$(echo "$line" | awk '{print $1}' | tr -d ' ')
                [ "$mac_c" = "No" ] && continue

				signal_c_dbm=$(echo "$line" | awk '{print $2}' | tr -d ' ')

				if [ "${VR_MACS[$mac_c]}" = "1" ]; then
                    is_vr=true
                else
                    is_vr=false
                fi

                ip_c=$(ip neigh | grep -i "$mac_c" | awk '$6 == "REACHABLE" || $6 == "STALE" {print $1}' | head -n 1)
                [ -z "$ip_c" ] && ip_c="N/A"

				iw_details=$(iw dev "$interface" station dump | grep -A 32 -i "$mac_c")

				# Extract RX/TX bytes and duration + TX retries and packets
				bytes_rx=$(echo "$iw_details" | grep "rx bytes" | awk '{print $3}')
				bytes_tx=$(echo "$iw_details" | grep "tx bytes" | awk '{print $3}')
				
				retries_tx=$(echo "$iw_details" | grep "tx retries" | awk '{print $(NF)}' | tr -d ' ')
				pkts_tx=$(echo "$iw_details" | grep "tx packets" | awk '{print $(NF)}' | tr -d ' ')

				duration_rx=$(echo "$iw_details" | grep "rx duration" | awk '{print $3}' | tr -d 'us')
				duration_tx=$(echo "$iw_details" | grep "tx duration" | awk '{print $3}' | tr -d 'us')

				current_time_ms=$(echo "$iw_details" | grep "current time" | awk '{print $(NF-1)}')

				mcs_tx="N/A"
            fi
            if echo "$line" | grep -Eq 'TX:'; then
				tx_details=$(echo "$line" | grep "TX:")
                if echo "$tx_details" | grep -Eq 'MCS'; then
					mcs_tx=$(echo "$tx_details" | awk '{print $5}' | tr -d ' ,')
                else
					mcs_tx="N/A"
				fi
                # Append client details to JSON
				c_json=$(cat <<EOF
				{	"ip": "$ip_c",
					"mac": "$mac_c",
					"is_vr": "$is_vr",
					"signal_dbm": "$signal_c_dbm",
					"current_time_ms": "$current_time_ms",
					"tx": {
						"mcs": "$mcs_tx",
						"bytes": "$bytes_tx",
						"packets": "$pkts_tx",
						"retries": "$retries_tx",
						"duration": "$duration_tx"
					},
					"rx": {
						"bytes": "$bytes_rx",
						"duration": "$duration_rx"
					}
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
			"ch_active_time_ms": "$ch_active_time_ms",
			"ch_busy_time_ms": "$ch_busy_time_ms",
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