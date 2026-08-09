# Fleet WiFi

The qualifier fleet uses one field SSID on the onboard `wlan0` radio. One
drone has the `master` role and hosts the access point. Every other drone has
the `client` role and joins it. The role comes from `/etc/maav/fleet-role`.

The network uses NetworkManager AP mode. NetworkManager also owns the client
profiles and their priority. This avoids a handoff between NetworkManager,
`hostapd`, and `wpa_supplicant`. A separate `dnsmasq` process gives addresses
to operator devices. It does not provide DNS, a default route, forwarding, or
NAT.

The old `wifi_poc` script and `maav-wifi-poc.service` are replaced by the image
overlay and provisioning files in `px4_ros_build`.

## Address and role model

| Host | Fleet address on `wlan0` | Initial role | PX4 namespace |
| --- | --- | --- | --- |
| `drone0` | `10.77.0.1/24` | `master` | `px4_0` |
| `drone1` | `10.77.0.11/24` | `client` | `px4_1` |
| `drone2` | `10.77.0.12/24` | `client` | `px4_2` |
| `drone3` | `10.77.0.13/24` | `client` | `px4_3` |

The address belongs to the drone. A role change does not change any address.
The master can therefore host the access point from `.11`, `.12`, or `.13`.
The CycloneDDS peer list remains valid after a role change.

The field SSID, field PSK, channel, DHCP range, operator hotspot, and static
operator-device leases come from `px4_ros_build/provision/inventory.yml`.
Drone addresses and roles are in the `qualifier_fleet` host entries. The drones
use static addresses. They do not use the DHCP range or static DHCP leases.

## Boot behavior

`maav-fleet-network.service` reads these provisioned files:

- `/etc/maav/fleet-role` contains `master` or `client`.
- `/etc/maav/fleet-network.conf` contains the common WiFi settings and the
  drone's fixed fleet address.
- `/etc/maav/dnsmasq-fleet.conf` contains the operator-device DHCP settings.

The master activates `maav-field-ap` on `wlan0` with its own fleet address.
`maav-fleet-dhcp.service` then starts `dnsmasq` on that interface.

A client creates `maav-field-client` with its own fleet address. NetworkManager
keeps retrying the profile if no known SSID is in range. This condition does not
fail the fleet network unit. ROS can start and use loopback DDS while the radio
is disconnected.

Client autoconnect priority is:

| Priority | Network | Address source |
| --- | --- | --- |
| 300 | `MWireless` | DHCP |
| 200 | Operator hotspot from inventory | DHCP |
| 100 | Fleet field SSID | Static `fleet_ip` |

NetworkManager selects the highest available priority when it needs a
connection. It does not leave an active field connection when a higher-priority
SSID later appears.

If the master has a USB WiFi adapter at service start, the service creates
adapter-specific MWireless and hotspot profiles. The onboard radio remains the
field access point. The USB adapter is optional and is not part of the flight
configuration. Restart `maav-fleet-network.service` after inserting an adapter
that was not present at boot.

## Provisioning

The image includes the controller, systemd units, and `dnsmasq`. `flash.nu`
writes the durable development WiFi source at
`/usr/lib/netplan/50-maav-wifi.yaml`. For a qualifier host, it also writes the
role, address, and DHCP files from inventory.

Apply or update the same configuration on reachable drones with:

```bash
cd px4_ros_build/provision
ansible-playbook playbooks/fleet-network.yml -K
```

The playbook installs the files, applies netplan, and restarts the role service.
Use `--limit` when only selected drones must change.

## Field operations

### Switch to flight mode

Turn off the operator hotspot or move it out of range. No drone command is
required. A disconnected client joins the field SSID when the master access
point becomes available.

If a client is already connected to the operator hotspot, turn the hotspot off.
NetworkManager then selects the field profile.

### Return to development mode

Turn on MWireless or the operator hotspot. NetworkManager does not roam away
from an active field connection only because a higher-priority SSID appears.
Run the reconnect nudge from the operator laptop:

```bash
cd px4_ros_build/provision
ansible-playbook playbooks/wifi-dev-reconnect.yml -K
```

The playbook schedules the reconnect and returns before SSH can move to the new
network. Each client then asks NetworkManager to select the best visible
profile. The master continues to host the field SSID on `wlan0`; its optional
USB adapter selects the development network.

### Change the master

Change `fleet_role` to `client` for the old master and to `master` for the new
master in `px4_ros_build/provision/inventory.yml`. Apply the client flag to the
old master first. Then apply the master flag to the new master:

```bash
cd px4_ros_build/provision
ansible-playbook playbooks/fleet-network.yml --limit old-master -K
ansible-playbook playbooks/fleet-network.yml --limit new-master -K
```

Do this while the drones also have a development or wired management path. This
order stops the old AP before the new AP starts. Restart
`maav-fleet-network.service` on the old master first and the new master second if
the files are changed by hand. There must be one active master. The `fleet_ip`,
PX4 namespace, and CycloneDDS peer entries do not change.

## Checks

Check one provisioned drone with:

```bash
sudo /usr/local/sbin/maav-fleet-network check
systemctl status maav-fleet-network.service maav-fleet-dhcp.service
nmcli -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show
ip -4 address show dev wlan0
```

Run the controller test without WiFi hardware with:

```bash
mission10/network/test_fleet_network.sh
```

Hardware tests at M-Air must confirm:

- the CM5 onboard radio supports the selected NetworkManager AP settings;
- all four drones associate and retain their inventory addresses after boot;
- a phone and laptop receive dynamic or MAC-keyed leases;
- MWireless, hotspot, and field fallback order works after each reconnect;
- a client with no known SSID keeps retrying while ROS stays healthy;
- the master's USB adapter joins each development network without changing the
  onboard AP;
- moving the master role between two drones stops the old AP before the new AP
  starts and does not change either address;
- fleet DDS discovery and operator access work at the required field range.
