#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly REPO_ROOT
readonly CONTROLLER="$REPO_ROOT/px4_ros_build/overlay/usr/local/sbin/fleet-network"
TEST_ROOT="$(mktemp -d)"
readonly TEST_ROOT
readonly TEST_BIN="$TEST_ROOT/bin"
readonly TEST_CONFIG="$TEST_ROOT/fleet-network.conf"
readonly TEST_MODE="$TEST_ROOT/fleet-mode"
readonly NMCLI_LOG="$TEST_ROOT/nmcli.log"
readonly SYSTEMCTL_LOG="$TEST_ROOT/systemctl.log"
readonly TEST_DDS="$TEST_ROOT/cyclonedds"
readonly MWIRELESS_UUID="00000000-0000-0000-0000-000000000001"
readonly HOTSPOT_UUID="00000000-0000-0000-0000-000000000002"
readonly VENUE_UUID="00000000-0000-0000-0000-000000000003"
readonly FIELD_SSID_UUID="00000000-0000-0000-0000-000000000004"
export NMCLI_LOG SYSTEMCTL_LOG MWIRELESS_UUID HOTSPOT_UUID VENUE_UUID FIELD_SSID_UUID

cleanup() {
    rm -rf -- "$TEST_ROOT"
}
trap cleanup EXIT
mkdir -p "$TEST_BIN"
mkdir -p "$TEST_DDS"
touch "$TEST_DDS/internet.xml" "$TEST_DDS/field.xml"

cat >"$TEST_BIN/nmcli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >>"$NMCLI_LOG"
printf '\n' >>"$NMCLI_LOG"
if [[ "$*" == "-g connection.id connection show "* ]]; then
    profile="${*: -1}"
    case ",${NMCLI_EXISTING_PROFILES:-${NMCLI_EXISTING_PROFILE:-}}," in
        *",$profile,"*) exit 0 ;;
        *) exit 1 ;;
    esac
fi
case "$*" in
    "-g UUID connection show") printf '%s\n%s\n%s\n%s\n' \
        "$MWIRELESS_UUID" "$HOTSPOT_UUID" "$VENUE_UUID" "$FIELD_SSID_UUID" ;;
    "-g connection.type connection show field-ap"|\
    "-g connection.type connection show field-client") printf '802-11-wireless\n' ;;
    "-g 802-11-wireless.ssid connection show $MWIRELESS_UUID") printf 'MWireless\n' ;;
    "-g 802-11-wireless.ssid connection show $HOTSPOT_UUID") printf 'Test-Hotspot\n' ;;
    "-g 802-11-wireless.ssid connection show $VENUE_UUID") printf 'Test-Venue\n' ;;
    "-g 802-11-wireless.ssid connection show $FIELD_SSID_UUID") printf 'Test-Fleet\n' ;;
    "-g connection.interface-name connection show $MWIRELESS_UUID"|\
    "-g connection.interface-name connection show $HOTSPOT_UUID"|\
    "-g connection.interface-name connection show $VENUE_UUID"|\
    "-g connection.interface-name connection show $FIELD_SSID_UUID") printf 'wlan0\n' ;;
    "-g connection.type connection show $MWIRELESS_UUID"|\
    "-g connection.type connection show $HOTSPOT_UUID"|\
    "-g connection.type connection show $VENUE_UUID"|\
    "-g connection.type connection show $FIELD_SSID_UUID") printf '802-11-wireless\n' ;;
    "-g 802-11-wireless.mode connection show $MWIRELESS_UUID"|\
    "-g 802-11-wireless.mode connection show $HOTSPOT_UUID"|\
    "-g 802-11-wireless.mode connection show $VENUE_UUID"|\
    "-g 802-11-wireless.mode connection show $FIELD_SSID_UUID") printf 'infrastructure\n' ;;
    "-g SSID device wifi list ifname wlan0")
        printf 'MWireless\nTest-Hotspot\nTest-Venue\n'
        ;;
    "--wait 15 connection up "*)
        uuid="${5:-}"
        case ",${NMCLI_FAIL_UUIDS:-}," in
            *",$uuid,"*) exit 1 ;;
        esac
        ;;
    "--terse --fields DEVICE,TYPE device status") printf 'wlan0:wifi\nwlan1:wifi\n' ;;
    "-g GENERAL.CONNECTION device show wlan0") printf 'field-client\n' ;;
esac
EOF
chmod 0755 "$TEST_BIN/nmcli"

cat >"$TEST_BIN/systemctl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$SYSTEMCTL_LOG"
if [[ "$*" == "is-enabled --quiet jarvis-web.service" ]]; then
    [[ "${JARVIS_ENABLED:-no}" == yes ]]
fi
EOF
chmod 0755 "$TEST_BIN/systemctl"

cat >"$TEST_CONFIG" <<'EOF'
FLEET_INDEX=2
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
        FLEET_CONFIG_FILE="$TEST_CONFIG" \
        FLEET_MODE_FILE="$TEST_MODE" \
        FLEET_DDS_CONFIG_DIR="$TEST_DDS" \
        bash "$CONTROLLER" "$@"
}

assert_log() {
    grep -Fq -- "$1" "$NMCLI_LOG" || {
        echo "Missing nmcli call: $1" >&2
        exit 1
    }
}

reject_log() {
    if grep -Fq -- "$1" "$NMCLI_LOG"; then
        echo "Unexpected nmcli call: $1" >&2
        exit 1
    fi
}

assert_systemctl() {
    grep -Fq -- "$1" "$SYSTEMCTL_LOG" || {
        echo "Missing systemctl call: $1" >&2
        exit 1
    }
}

run_controller check | grep -Fq 'fleet_index=2 master=none network=internet'
: >"$NMCLI_LOG"
: >"$SYSTEMCTL_LOG"
NMCLI_EXISTING_PROFILE=field-ap run_controller apply
assert_log 'connection modify field-ap connection.autoconnect no'
assert_log "--wait 15 connection up $MWIRELESS_UUID ifname wlan0"
assert_log 'con-name usb-mwireless-wlan1'
assert_systemctl 'stop jarvis-web.service'
[[ "$(readlink "$TEST_DDS/active.xml")" == internet.xml ]]

: >"$NMCLI_LOG"
NMCLI_FAIL_UUIDS="$MWIRELESS_UUID,$HOTSPOT_UUID" run_controller apply
assert_log "--wait 15 connection up $VENUE_UUID ifname wlan0"
reject_log "connection up $FIELD_SSID_UUID ifname wlan0"
assert_log "connection modify $VENUE_UUID connection.autoconnect yes connection.autoconnect-priority 100"
assert_log "connection modify $FIELD_SSID_UUID connection.autoconnect no"

: >"$NMCLI_LOG"
: >"$SYSTEMCTL_LOG"
NMCLI_EXISTING_PROFILE=field-client run_controller set-mode 2 field
run_controller is-master
assert_log 'con-name field-ap'
assert_log 'connection modify field-client connection.autoconnect no'
assert_log '802-11-wireless.mode ap'
assert_log 'connection up field-ap ifname wlan0'
assert_systemctl 'start --no-block fleet-dhcp.service'
assert_systemctl 'stop jarvis-web.service'
[[ "$(readlink "$TEST_DDS/active.xml")" == field.xml ]]
reject_log 'con-name usb-mwireless-wlan1'

: >"$SYSTEMCTL_LOG"
JARVIS_ENABLED=yes NMCLI_EXISTING_PROFILE=field-client run_controller set-mode 2 field
assert_systemctl 'start --no-block jarvis-web.service'

: >"$NMCLI_LOG"
: >"$SYSTEMCTL_LOG"
run_controller set-mode 1 field
assert_log 'con-name field-client'
assert_log 'connection up field-client ifname wlan0'
assert_log 'ipv4.addresses 10.77.0.12/24'
assert_systemctl 'stop fleet-dhcp.service'
[[ "$(readlink "$TEST_DDS/active.xml")" == field.xml ]]

: >"$NMCLI_LOG"
: >"$SYSTEMCTL_LOG"
NMCLI_EXISTING_PROFILES=field-ap,field-client run_controller set-mode 2 internet
assert_log 'connection modify field-client connection.autoconnect no'
assert_log 'connection down field-client'
assert_log 'connection modify field-ap connection.autoconnect no'
assert_log 'connection down field-ap'
assert_log "--wait 15 connection up $MWIRELESS_UUID ifname wlan0"
reject_log 'connection up field-ap ifname wlan0'
reject_log 'connection.autoconnect-priority'
assert_systemctl 'stop jarvis-web.service'
assert_systemctl 'stop fleet-dhcp.service'
[[ "$(readlink "$TEST_DDS/active.xml")" == internet.xml ]]
if run_controller is-field-master; then
    echo "Internet mode was reported as a field master" >&2
    exit 1
fi

echo "fleet network controller tests passed"
