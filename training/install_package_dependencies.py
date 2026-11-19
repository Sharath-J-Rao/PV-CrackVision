"""Install metadata-declared dependencies without replacing protected packages.

Packages themselves must already be installed as wheels. This helper reads
Requires-Dist from their installed metadata, evaluates environment markers,
and resolves only those requirements from PyPI. Protected packages are pinned
at their currently installed versions via a temporary constraints file.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from importlib import metadata

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.markers import default_environment

PYPI_INDEX_URL = "https://pypi.org/simple"


def isolated_pip_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Ignore inherited indexes/find-links and machine pip config for deterministic resolution."""
    environment = (os.environ if source is None else source).copy()
    for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS"):
        environment.pop(name, None)
    environment["PIP_CONFIG_FILE"] = os.devnull
    return environment


def collect_requirements(package_names: list[str], excluded: set[str]) -> list[str]:
    environment = default_environment()
    # PEP 508 defines ``extra`` only while evaluating an optional extra.
    # For the base installation requirements, use the empty extra value.
    # Without it, packaging raises UndefinedEnvironmentName for metadata
    # entries such as ``some-package; extra == 'gpu'``.
    environment["extra"] = ""
    requirements: list[str] = []
    seen: set[str] = set()
    for package_name in package_names:
        try:
            raw_requirements = metadata.requires(package_name) or []
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Cannot resolve dependencies: {package_name!r} is not installed."
            ) from exc
        for raw in raw_requirements:
            requirement = Requirement(raw)
            normalized = canonicalize_name(requirement.name)
            if normalized in excluded:
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            rendered = str(requirement)
            if rendered not in seen:
                seen.add(rendered)
                requirements.append(rendered)
    return requirements


def installed_constraints(package_names: list[str]) -> list[str]:
    constraints: list[str] = []
    for package_name in package_names:
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Cannot protect {package_name!r}: it is not installed."
            ) from exc
        constraints.append(f"{canonicalize_name(package_name)}=={version}")
    return constraints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packages", nargs="+", help="Installed distributions whose dependencies should be resolved")
    parser.add_argument("--preserve", nargs="*", default=[], help="Installed distributions that pip must not replace")
    args = parser.parse_args()

    excluded = {canonicalize_name(name) for name in args.preserve}
    requirements = collect_requirements(args.packages, excluded)
    constraints = installed_constraints(args.preserve)
    if not requirements:
        print("No additional package dependencies are required.")
        return 0

    # Explicit PyPI selection is authoritative. Ignore inherited pip environment
    # and pip.ini/pip.conf so a machine-level extra index cannot change resolution.
    environment = isolated_pip_environment()

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix="-constraints.txt", delete=False) as stream:
        stream.write("\n".join(constraints))
        stream.write("\n")
        constraint_path = stream.name
    try:
        command = [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check",
            "--index-url", PYPI_INDEX_URL,
            "--constraint", constraint_path,
            *requirements,
        ]
        print("Resolving installed package requirements from PyPI:")
        for requirement in requirements:
            print(f"  - {requirement}")
        completed = subprocess.run(command, env=environment, check=False)
        return completed.returncode
    finally:
        try:
            os.unlink(constraint_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
