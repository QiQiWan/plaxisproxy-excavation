"""Read-only PLAXIS 3D V22 Input API probe. No modelling or calculation.

Example:
  PLAXIS_PASSWORD=... python scripts/probe_plaxis_v22.py --port 10000
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from itertools import islice


def _value(value):
    for _ in range(2):
        candidate = getattr(value, "value", value)
        if candidate is value:
            break
        value = candidate
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _read_field(obj, name):
    try:
        return _value(getattr(obj, name))
    except Exception as exc:
        return {"read_error": type(exc).__name__}


def inspect_input(global_input, limit=25):
    """Inspect phase metadata without changing the model or solving."""
    report = {"phases": [], "phase_read_error": None}
    try:
        phases = iter(global_input.Phases)
        for phase in islice(phases, limit):
            report["phases"].append({
                "identification": _read_field(phase, "Identification"),
                "calculation_result": _read_field(phase, "CalculationResult"),
                "log_info": _read_field(phase, "LogInfo"),
                "should_calculate": _read_field(phase, "ShouldCalculate"),
            })
    except Exception as exc:
        report["phase_read_error"] = type(exc).__name__
    report["inspected_phase_count"] = len(report["phases"])
    report["phase_count_truncated_at"] = limit
    return report


def run_probe(host, port, password, new_server):
    if not password:
        raise ValueError("Set PLAXIS_PASSWORD before connecting")
    server, global_input = new_server(host, port, password=password)
    try:
        return inspect_input(global_input)
    finally:
        close = getattr(server, "close", None)
        if callable(close):
            close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--scripting-path", type=Path, default=None,
                        help="PLAXIS V22 bundled Python site-packages directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.scripting_path is not None:
        if not args.scripting_path.is_dir():
            parser.error("--scripting-path must be an existing directory")
        sys.path.insert(0, str(args.scripting_path.resolve()))
    password = os.environ.get("PLAXIS_PASSWORD", "")
    if not password:
        parser.error("PLAXIS_PASSWORD must be provided via environment variable")

    try:
        pkg = importlib.import_module("plxscripting")
        easy = importlib.import_module("plxscripting.easy")
        client_path = getattr(pkg, "__file__", "unknown")
        report = run_probe(args.host, args.port, password, easy.new_server)
    except Exception as exc:
        report = {"connection_error_type": type(exc).__name__,
                  "connection_error": "Check the bundled scripting client and Input server"}
        result_code = 2
    else:
        result_code = 0
    report.update({
        "target": "PLAXIS 3D CONNECT Edition V22 / 22.1.0.452",
        "python": sys.version.split()[0],
        "client_module_path": str(locals().get("client_path", "not-imported")),
        "read_only": True,
    })
    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
