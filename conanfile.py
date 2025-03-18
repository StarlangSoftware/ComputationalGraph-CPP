from conan import ConanFile
from conan.tools.cmake import CMakeToolchain


class ComputationalGraphConan(ConanFile):
    name = "ComputationalGraph"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    requires = "math/1.0.0"

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
