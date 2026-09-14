def resolution_status(
    pixel_width_m: float | None,
    pixel_height_m: float | None,
    warn_threshold_m: float = 1.0,
    block_threshold_m: float = 5.0,
) -> tuple[str, list[str]]:
    if pixel_width_m is None or pixel_height_m is None:
        return "unknown", ["Ground sample distance could not be established."]

    gsd = max(pixel_width_m, pixel_height_m)

    if gsd >= block_threshold_m:
        return "blocked", [
            f"Spatial resolution is about {gsd:.2f} m/pixel; "
            "individual crowns may not be reliably separable."
        ]

    if gsd >= warn_threshold_m:
        return "warning", [
            f"Spatial resolution is about {gsd:.2f} m/pixel; "
            "crown-scale detection may be less reliable."
        ]

    return "good", []
