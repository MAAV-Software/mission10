"""Process-local libcamera tuning helpers for daylight camera capture."""

def cap_short_exposure(tuning, max_exposure_us, camera_name):
    """Cap camera AGC channel zero's short-mode shutter curve.

    The curve remains automatic below the limit. Once the shutter reaches the
    limit, AGC increases analogue gain along the remaining curve points.
    """
    if max_exposure_us <= 0:
        raise ValueError("maximum exposure must be positive")

    for algorithm in tuning.get("algorithms", []):
        agc = algorithm.get("rpi.agc")
        if not agc:
            continue
        channels = agc.get("channels", [])
        agc_channel = channels[0] if channels else agc
        exposure_modes = agc_channel.get("exposure_modes", {})
        short_mode = exposure_modes.get("short")
        if not short_mode or not short_mode.get("shutter"):
            break
        short_mode["shutter"] = [
            min(int(shutter_us), int(max_exposure_us))
            for shutter_us in short_mode["shutter"]
        ]
        return tuning

    raise RuntimeError(
        f"{camera_name} tuning does not contain "
        "rpi.agc channel 0 short exposure mode"
    )


def cap_imx219_short_exposure(tuning, max_exposure_us):
    return cap_short_exposure(tuning, max_exposure_us, "IMX219")


def load_imx219_daylight_tuning(picamera2_class, max_exposure_us):
    """Load the installed IMX219 tuning and apply a hard shutter ceiling."""
    tuning = picamera2_class.load_tuning_file("imx219.json")
    return cap_imx219_short_exposure(tuning, max_exposure_us)


def load_ov9281_daylight_tuning(picamera2_class, max_exposure_us):
    """Load the installed OV9281 tuning and apply a hard shutter ceiling."""
    tuning = picamera2_class.load_tuning_file("ov9281_mono.json")
    return cap_short_exposure(tuning, max_exposure_us, "OV9281")
