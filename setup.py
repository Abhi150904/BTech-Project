"""Copy only reviewed, byte-preserved runtime resources into wheel builds."""

import runpy
from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
RESOURCES = runpy.run_path(str(ROOT / "src/runtime_security/assets.py"))["RESOURCE_FILES"]


class BuildWithResources(build_py):
    def run(self):
        super().run()
        for relative in RESOURCES:
            destination = Path(self.build_lib) / "runtime_security/_data" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(ROOT / relative, destination)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "runtime_security/_data" / relative)
            for relative in RESOURCES
        ]


setup(cmdclass={"build_py": BuildWithResources})
