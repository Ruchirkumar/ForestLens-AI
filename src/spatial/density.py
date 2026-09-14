def tree_density(tree_count: int, area_ha: float | None) -> float | None:
    if area_ha is None or area_ha <= 0:
        return None

    return tree_count / area_ha
