"""Temporary: find which earlier test leaves the next mobafit call hung."""

import subprocess
import sys
import time

FIT = "tests/test_cli.py::test_mobafit_writes_the_commands_coefficients_to_the_tolerance_of_a_fit"
ROUTE = "tests/test_cli.py::test_a_mobafit_the_app_cannot_express_goes_to_the_command"
MAG = "tests/test_cli.py::test_a_magnitude_mobafit_writes_the_commands_coefficients"
MODELS = ["D", "G0", "G0 from a start", "G1", "G1 from a start", "G3", "G3 from a start",
          "G4", "I", "L", "M", "T"]

CASES = [[MAG]] + [[f"{FIT}[{m}]", MAG] for m in MODELS] + [[ROUTE, MAG], [FIT, MAG]]

for case in CASES:
    start = time.time()
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
             "-o", "faulthandler_timeout=90", *case],
            capture_output=True, text=True, timeout=180,
        )
        verdict, out = f"exit {done.returncode}", done.stdout + done.stderr
    except subprocess.TimeoutExpired as hung:
        verdict = "HUNG"
        out = (hung.stdout or b"").decode(errors="replace") if isinstance(hung.stdout, bytes) else (hung.stdout or "")
        out += (hung.stderr or b"").decode(errors="replace") if isinstance(hung.stderr, bytes) else (hung.stderr or "")
    print(f"=== {verdict} after {time.time() - start:.0f} s: {case}", flush=True)
    if verdict != "exit 0":
        print("\n".join(out.splitlines()[-120:]), flush=True)
