unit_to_meters = {
    'm': 1.,
    'cm': .01,
    'mm': .001,
    'in': .0254,
    'inch': .0254,
    'inches': .0254,
    'ft': .3048,
    'foot': .3048,
    'feet': .3048,
}
SUPPORTED_UNITS = tuple(unit_to_meters)


def normalize_unit(unit: str | None) -> str:
    return unit.strip().lower() if isinstance(unit, str) else ''


def is_supported_unit(unit: str | None) -> bool:
    return normalize_unit(unit) in unit_to_meters


def conversion_factor(source_unit: str, target_unit: str) -> float:
    normalized_source = normalize_unit(source_unit)
    normalized_target = normalize_unit(target_unit)
    if normalized_source not in unit_to_meters:
        raise ValueError(f'Unsupported source unit {source_unit!r}; expected one of: {", ".join(SUPPORTED_UNITS)}')
    if normalized_target not in unit_to_meters:
        raise ValueError(f'Unsupported target unit {target_unit!r}; expected one of: {", ".join(SUPPORTED_UNITS)}')
    return unit_to_meters[normalized_source] / unit_to_meters[normalized_target]