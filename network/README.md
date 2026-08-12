# Fleet WiFi

Every qualifier drone has one required radio: the CM5 `wlan0`. One optional USB
adapter can be connected to any drone as `wlan1`. The adapter provides a
development connection for that drone only. The fleet does not use forwarding
or NAT.

UWB carries the complete runtime selection:

```text
FleetMode { master: uint8, network: field | internet }
```

All receivers persist the last packet and apply it. In `field` mode, the
selected master's `wlan0` is the fleet access point. In `internet` mode, every
`wlan0` joins the best development network. There is no compiled or provisioned
master. A drone with no saved mode joins a development network. Jarvis remains
stopped until its runtime assets are installed and `jarvis-web.service` is
enabled.

## Addresses

| Host | Fleet address on `wlan0` | PX4 namespace |
| --- | --- | --- |
| `drone0` | `10.77.0.10/24` | `px4_0` |
| `drone1` | `10.77.0.11/24` | `px4_1` |
| `drone3` | `10.77.0.13/24` | `px4_3` |
| `drone4` | `10.77.0.14/24` | `px4_4` |

The address belongs to the drone. A master change does not change any address.
The CycloneDDS peer list therefore remains valid.

The field LAN has no default gateway, forwarding, or NAT. Peers in the same
`10.77.0.0/24` subnet communicate directly; `10.77.0.1` is intentionally
unassigned.

Addresses `10.77.0.10` through `10.77.0.19` are reserved for companion
computers. Known drones use `.10 + fleet_index`; unused addresses remain
available for replacement Pis. Operator-device DHCP uses `.100` through `.199`.

## Runtime state

`fleet-network.service` uses:

- `/etc/maav/fleet-network.conf` for `FLEET_INDEX`, addresses, and credentials;
- `/var/lib/maav/fleet-mode` for the last received master and network;
- `/etc/maav/dnsmasq-fleet.conf` for operator-device DHCP in field mode.

Set and apply the mode directly with:

```bash
sudo fleet-network set-mode 2 field
sudo fleet-network set-mode 0 internet
```

The operator command shows the current role, radios, and related services. Its
two-argument form broadcasts the selection over UWB; `local` applies it only to
the current drone:

```bash
fleet status
fleet 0 field
fleet local 0 internet
```

`fleetmode MASTER field|internet` remains available for compatibility.

The ROS service `/<namespace>/uwb/set_fleet_mode` sends the same mode eight times over UWB
and applies it locally. A DWM3001 host can send the same packet without ROS.

In field mode, the selected master activates `field-ap` on `wlan0` and starts
`dnsmasq`. Followers activate `field-client` with their fixed fleet addresses.
In internet mode, all drones prefer MWireless and then the configured hotspot.
If neither connects, they try operator-saved infrastructure profiles bound to
`wlan0`, such as venue Wi-Fi. The field SSID and profiles for other interfaces
are excluded. DHCP is stopped. When `jarvis-web.service` is enabled, only the
selected master runs Jarvis; a disabled service stays stopped in both modes.

Every image contains dormant, interface-specific MWireless and hotspot profiles
for `wlan1`. NetworkManager uses them when the one USB adapter is inserted.
Changing `wlan0` does not disconnect a working `wlan1` management path.
Venue profiles use autoconnect priority 100, below MWireless (300), the
configured hotspot (200), and the active field role (400).

## Provisioning

The image and `px4_ros_build/provision/playbooks/fleet-network.yml` install the
controller, systemd units, durable development credentials, fleet addresses,
and USB profiles.

Apply the configuration with:

```bash
cd px4_ros_build/provision
ansible-playbook playbooks/fleet-network.yml -K
```

## Checks

```bash
sudo fleet-network check
systemctl status fleet-network.service fleet-dhcp.service jarvis-web.service
nmcli -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show
ip -4 address show dev wlan0
```

Run the hardware-free controller test with:

```bash
mission10/network/test_fleet_network.sh
```

Before flight, confirm that one `FleetMode` packet changes all four drones, only
the selected field master runs DHCP, `wlan1` remains connected during `wlan0`
changes, fleet DDS discovery works, and a late-starting drone applies the next
repeated mode packet.
