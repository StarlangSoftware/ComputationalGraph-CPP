# Getting Started and Verification Instructions

This guide provides step-by-step instructions for cloning the `ComputationalGraph-CPP` project and setting it up with the required dependencies from the Ozyegin remote.

## 1. Prerequisites
Ensure you have the following installed on your system:
- **C++ Compiler** (supporting C++20)
- **CMake** (v3.12 or higher)
- **Python 3** and **pip**
- **Conan 2.x**

To install Conan:
```bash
pip install conan
```

## 2. Clone the Repository
Clone the project from GitHub:
```bash
git clone https://github.com/StarlangSoftware/ComputationalGraph-CPP.git
cd ComputationalGraph-CPP
```

## 3. Configure Conan Remote
Add the Ozyegin remote to your Conan configuration:
```bash
conan remote add ozyegin http://104.247.163.162:8081/artifactory/api/conan/conan-local
```

## 4. Detect default profile (if not already done)
```bash
conan profile detect --force
```

## 5. Install Dependencies
Install the required packages (`math`, `classification`, etc.) and generate the build files:
```bash
conan install . --build=missing
```

## 6. Build the Project
Configure and compile the library and the test executable using CMake:
```bash
# Configure with the generated Conan toolchain
cmake -S . -B build/Release -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build/Release
```

## 7. Run Verification Tests
Execute the `CGTest` file to verify the project is working correctly with the remote dependencies:
```bash
./build/Release/CGTest
```

**Expected Output:**
If the setup is successful, you should see the following in your terminal:
```text
Logarithm Test Passed!
Power Test Passed!
SiLU Test Passed!
TanhShrink Test Passed!
Linear Perceptron Test Finished!
```
