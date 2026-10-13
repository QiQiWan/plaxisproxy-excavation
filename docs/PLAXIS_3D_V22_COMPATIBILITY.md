# PLAXIS 3D V22 (22.1.0.452) API compatibility

This repository is used with a PLAXIS 3D CONNECT Edition V22
installation for excavation-tunnel-servo research. Do not assume that a
newer scripting client is backward compatible.

## Version constraint

The existing pyproject.toml declares plxscripting>=1.0.4. The published
compatibility matrix for plxscripting 1.0.4 identifies PLAXIS 3D 2023.2,
not PLAXIS 3D V22. That dependency must not be treated as V22-certified.

Use the plxscripting client shipped with the licensed V22 installation,
and capture its path and Python version. Do not copy vendor libraries
into this repository or distribute proprietary PLAXIS Python files.

## Read-only API inspection

Run the probe against an existing Input remote scripting service. Set
PLAXIS_PASSWORD in the environment; do not commit a password to source.

PowerShell example (replace the client path with the real installation):

~~~powershell
$env:PLAXIS_PASSWORD = '<your own scripting password>'
python scripts/probe_plaxis_v22.py --host 127.0.0.1 --port 10000 --scripting-path 'C:\PATH\TO\PLAXIS\python\Lib\site-packages' --output artifacts/v22_input_api.json
~~~

The probe reads Identification, CalculationResult, LogInfo, and
ShouldCalculate for available phases. It does not create a model,
change parameters, calculate, or connect to Output. In an empty project,
having no staged phases is not by itself a solver failure.

Run tests without PLAXIS:

~~~shell
python -m unittest discover -s tests -p test_plaxis_v22_probe.py -v
~~~

## V22 integration gates

| Capability | Minimum live evidence |
| --- | --- |
| Input connection | new_server works with installed V22 client |
| Phase state | Phase CalculationResult and LogInfo read correctly |
| Phase overrides | P7 change does not affect P6 |
| Tunnel geometry | Circular lining, interface, soil void all created |
| Servo loading | Compression convention and actual reaction verified |
| Output | V22 Plate and anchor result field names verified |
| Data campaign | One full history passes all quality checks |

The probe tests a narrow, read-only Input contract. It is not a
substitute for a real tunnel, prestress or Output integration test.

References:
- Bentley: How to handle a calculation state with Python scripting.
- cemsbv/plxscripting compatibility table: 1.0.4 -> PLAXIS 3D 2023.2.

The packaging dependency should be corrected in a separate, reviewed
change after identifying the locally installed V22 scripting client.
