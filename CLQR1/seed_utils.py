def _parse_positive_int(value, field_name, source_text):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "invalid {0} in seed spec {1!r}: expected an integer.".format(field_name, source_text)
        )
    if parsed < 0:
        raise ValueError(
            "invalid {0} in seed spec {1!r}: expected a non-negative integer.".format(field_name, source_text)
        )
    return parsed


def parse_seed_list(seeds_text):
    text = "" if seeds_text is None else str(seeds_text).strip()
    if not text:
        return []

    seeds = []
    seen = set()
    for raw_seed in text.split(","):
        seed_text = raw_seed.strip()
        if not seed_text:
            continue
        seed_value = _parse_positive_int(seed_text, "seed", text)
        if seed_value in seen:
            continue
        seen.add(seed_value)
        seeds.append(seed_value)
    return seeds


def resolve_experiment_seeds(args, default_seed):
    seeds_text = getattr(args, "seeds", None)
    parsed_seeds = parse_seed_list(seeds_text)
    if parsed_seeds:
        return parsed_seeds
    return [int(getattr(args, "seed", default_seed))]
