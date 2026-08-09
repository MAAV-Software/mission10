# Comms-link telemetry a drone broadcasts to its peers (distinct from the UWB
# range sensor). phase/phase_rate drive the L1 phase-lock; yaw is the shared
# heading; position/velocity are the peer's own ENU estimate (used by L2 reflex).
builtin_interfaces/Time stamp
uint32 sequence
uint8 vehicle_id
uint16 frame_epoch
uint16 VALID_PHASE=1
uint16 VALID_YAW=2
uint16 VALID_POSITION=4
uint16 VALID_VELOCITY=8
uint16 HOST_STALE=32768
uint16 validity
float32 phase_rad
float32 phase_rate_rad_s
float32 yaw_rad
float32[3] position_enu_m
float32[3] velocity_enu_mps
# The drone's own absolute GNSS estimate (ENU). Inside the 8 m circle this is
# corrupted by metre-level common-mode error; a peer differences it against its
# own GNSS to recover a usable relative-position vector (differential GNSS).
float32[3] gnss_enu_m
uint8 MODE_NOMINAL=0
uint8 MODE_PHASE=1
uint8 MODE_REFLEX=2
uint8 MODE_DECONFLICT=3
uint8 mode
# Deconfliction diagnostics: predicted closest-approach to the peer and the
# resulting deconfliction severity (0 when no conflict). NaN t/d = peer unknown.
float32 t_cpa_s
float32 d_cpa_m
float32 deconflict_severity
