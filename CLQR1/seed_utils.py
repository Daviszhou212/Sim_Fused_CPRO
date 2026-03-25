import argparse
import sys


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


def _normalize_argv(argv):
    raw_args = sys.argv[1:] if argv is None else list(argv)
    return [str(item).strip() for item in raw_args if str(item).strip()]


def _dest_to_option_name(field_name):
    return "--{0}".format(str(field_name).replace("_", "-"))


def collect_explicit_cli_options(argv, protected_fields):
    normalized_args = _normalize_argv(argv)
    explicit_options = []
    for field_name in protected_fields:
        option_name = _dest_to_option_name(field_name)
        if any((item == option_name) or item.startswith(option_name + "=") for item in normalized_args):
            explicit_options.append(option_name)
    return explicit_options


def apply_python_config_priority(cli_args, python_config, protected_fields, argv=None):
    missing_fields = [field_name for field_name in protected_fields if field_name not in python_config]
    if missing_fields:
        raise KeyError(
            "python_config is missing protected fields: {0}".format(", ".join(str(field) for field in missing_fields))
        )

    merged_args = argparse.Namespace(**dict(python_config))
    for field_name, field_value in vars(cli_args).items():
        if field_name in protected_fields:
            continue
        setattr(merged_args, field_name, field_value)

    ignored_options = collect_explicit_cli_options(argv, protected_fields)
    return merged_args, ignored_options


def format_ignored_cli_overrides(ignored_options):
    normalized_options = [str(option).strip() for option in ignored_options if str(option).strip()]
    if not normalized_options:
        return ""
    return "Ignored CLI overrides in favor of Python config: {0}".format(", ".join(normalized_options))
