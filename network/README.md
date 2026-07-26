# Wi-Fi proof of concept

## Purpose

This test connects drone1 to an access point on drone2.

Drone2 uses its onboard Wi-Fi interface for the access point. Its USB Wi-Fi
adapter stays connected to MWireless.

Drone1 uses its onboard Wi-Fi interface as a station. Drone1 has the static
address `10.77.0.11`.

## Files

Copy these files to each drone:

```text
/home/maav/wifi_poc
/etc/systemd/system/maav-wifi-poc.service
```

The script contains all settings for this test. The test does not use a
separate configuration file.

## Start the test

1. Start the service on drone2.

   ```bash
   sudo systemctl start maav-wifi-poc.service
   ```

2. Start the service on drone1.

   ```bash
   sudo systemctl start maav-wifi-poc.service
   ```

3. Connect from drone2 to drone1.

   ```bash
   ssh drone1
   ```

## Show the status

Run this command on a drone:

```bash
sudo /home/maav/wifi_poc status
```

## Stop the test

Run this command on a drone:

```bash
sudo systemctl stop maav-wifi-poc.service
```

A reboot returns drone1 to MWireless. The proof-of-concept service is not
enabled at startup.
