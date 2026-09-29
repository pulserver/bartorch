"""Print the prefixes oneMKL's include/fftw may be under, separated by semicolons.

The distribution's own record comes first because inside pip's build isolation
the header is under an overlay that neither prefix of the interpreter names.
"""

import importlib.metadata
import sys
import sysconfig

prefixes = []
for name in ("mkl-include", "mkl-devel"):
    try:
        files = importlib.metadata.distribution(name).files or []
    except importlib.metadata.PackageNotFoundError:
        continue
    for file in files:
        if file.name == "fftw3_mkl.h":
            prefixes.append(str(file.locate().resolve().parent.parent.parent))
prefixes += [sysconfig.get_paths()["data"], sys.prefix]
print(";".join(dict.fromkeys(prefixes)))
