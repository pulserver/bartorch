"""Temporary: find which earlier test leaves the next mobafit call hung."""

import subprocess
import sys
import time

FIT = "tests/test_cli.py::test_mobafit_writes_the_commands_coefficients_to_the_tolerance_of_a_fit"
ROUTE = "tests/test_cli.py::test_a_mobafit_the_app_cannot_express_goes_to_the_command"
MAG = "tests/test_cli.py::test_a_magnitude_mobafit_writes_the_commands_coefficients"
MODELS = ["D", "G0", "G0 from a start", "G1", "G1 from a start", "G3", "G3 from a start",
          "G4", "I", "L", "M", "T"]

PREFIX = ["tests/test_abi.py", "tests/test_apps.py", "tests/test_build.py",
          "tests/test_catalogue.py", "tests/test_cfl.py"]
CLI = "tests/test_cli.py"
CASES = [
    [*PREFIX, CLI],
    ["tests/test_apps.py", CLI],
    ["tests/test_abi.py", "tests/test_catalogue.py", "tests/test_cfl.py", CLI],
    ["tests/test_apps.py", FIT, MAG],
    ["tests/test_apps.py", MAG],
    [CLI],
]

CHILD = (
    "import faulthandler, sys, pytest\n"
    "f = open('stacks.txt', 'w')\n"
    "faulthandler.dump_traceback_later(300, file=f)\n"
    "sys.path.insert(0, 'scripts')\n"
    "sys.exit(pytest.main(['-v', '-p', 'no:cacheprovider', '-p', 'no:faulthandler', '-p', '_memplugin', *sys.argv[1:]]))\n"
)

for case in CASES:
    start = time.time()
    child = subprocess.Popen([sys.executable, "-c", CHILD, *case], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = child.communicate(timeout=420)
        verdict = f"exit {child.returncode}"
    except subprocess.TimeoutExpired:
        verdict = "HUNG"
        spy = subprocess.run(["py-spy", "dump", "--native", "--locals", "--pid", str(child.pid)],
                             capture_output=True, text=True)
        print("--- py-spy ---\n" + spy.stdout + spy.stderr, flush=True)
        child.kill()
        out, _ = child.communicate()
    print(f"=== {verdict} after {time.time() - start:.0f} s: {case}", flush=True)
    try:
        print(open("mem.txt").read(), flush=True)
    except OSError:
        pass
    if verdict != "exit 0":
        print("\n".join(out.splitlines()[-120:]), flush=True)
        try:
            print("--- stacks ---\n" + open("stacks.txt").read(), flush=True)
        except OSError:
            pass
