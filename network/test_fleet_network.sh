#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly REPO_ROOT
readonly CONTROLLER="$REPO_ROOT/px4_ros_build/overlay/usr/local/sbin/maav-fleet-network"
TEST_ROOT="$(mktemp -d)"
readonly TEST_ROOT
readonly TEST_BIN="$TEST_ROOT/bin"
readonly TEST_CONFIG="$TEST_ROOT/fleet-network.conf"
readonly TEST_ROLE="$TEST_ROOT/fleet-role"
readonly NMCLI_LOG="$TEST_ROOT/nmcli.log"
export NMCLI_LOG

cleanup() {
    rm -rf -- "$TEST_ROOT"
}
trap cleanup EXIT

mkdir -p "$TEST_BIN"

cat >"$TEST_BIN/nmcli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%q ' "$@" >>"$NMCLI_LOG"
printf '\n' >>"$NMCLI_LOG"

if [[ "$*" == "-g connection.id connection show ${NMCLI_EXISTING_PROFILE:-__none__}" ]]; then
    exit 0
fi

case "$*" in
    "-g connection.id connection show "*)
        exit 1
        ;;
    "-g UUID connection show")
        exit 0
        ;;
    "--terse --fields DEVICE,TYPE device status")
        printf 'wlan0:wifi\nwlan1:wifi\n'
        ;;
    "-g GENERAL.CONNECTION device show wlan0")
        printf 'maav-field-client\n'
        ;;
esac
EOF
chmod 0755 "$TEST_BIN/nmcli"

cat >"$TEST_CONFIG" <<'EOF'
FIELD_SSID=Test-Fleet
FIELD_PSK=test-field-password
FIELD_CHANNEL=6
FLEET_IP=10.77.0.12
DEV_IDENTITY=tester@umich.edu
DEV_PASSWORD=test-enterprise-password
DEV_CA_CERTIFICATE=/etc/ssl/certs/test.pem
HOTSPOT_SSID=Test-Hotspot
HOTSPOT_PSK=test-hotspot-password
EOF

run_controller() {
    PATH="$TEST_BIN:$PATH" \
        MAAV_FLEET_CONFIG_FILE="$TEST_CONFIG" \
        MAAV_FLEET_ROLE_FILE="$TEST_ROLE" \
        bash "$CONTROLLER" "$@"
}

assert_log() {
    grep -Fq -- "$1" "$NMCLI_LOG" || {
        echo "Missing nmcli call: $1" >&2
        exit 1
    }
}

printf 'client\n' >"$TEST_ROLE"
run_controller check | grep -Fq 'role=client fleet_ip=10.77.0.12 field_ssid=Test-Fleet'
: >"$NMCLI_LOG"
NMCLI_EXISTING_PROFILE=maav-field-ap run_controller apply
assert_log 'con-name maav-field-client'
assert_log 'connection modify maav-field-ap connection.autoconnect no'
assert_log 'connection.autoconnect-priority 100'
assert_log 'ipv4.addresses 10.77.0.12/24'

printf 'master\n' >"$TEST_ROLE"
run_controller is-master
: >"$NMCLI_LOG"
NMCLI_EXISTING_PROFILE=maav-field-client run_controller apply
assert_log 'con-name maav-field-ap'
assert_log 'connection modify maav-field-client connection.autoconnect no'
assert_log '802-11-wireless.mode ap'
assert_log 'connection up maav-field-ap ifname wlan0'
assert_log 'con-name maav-usb-mwireless-wlan1'
assert_log 'con-name maav-usb-hotspot-wlan1'

printf 'invalid\n' >"$TEST_ROLE"
if run_controller check >/dev/null 2>&1; then
    echo "An invalid role passed validation" >&2
    exit 1
fi

echo "fleet network controller tests passed"
