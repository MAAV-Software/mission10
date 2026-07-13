# UWB physical measurement: scalar range between two nodes, from ground-truth
# geometry plus sensor noise/dropout. No telemetry here; a peer's phase,
# orientation and position-estimate ride the separate UwbState comms broadcast.
builtin_interfaces/Time stamp
uint32 sequence
uint8 source_id
uint8 receiver_id
float32 range_m
