English | [日本語](README.ja.md)

# MassMolKit

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![RDKit](https://img.shields.io/badge/RDKit-2024.03.5-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)

---

MassMolKit is a Python toolkit for handling molecules and mass spectrometry data.  
It provides utilities for canonical SMILES conversion, formula handling, and MS-related operations.  
It can be integrated as a Git submodule or installed in editable mode.

---

## 🔗 Add as a Git Submodule

To integrate MassMolKit into an existing project, run the following commands at the root of your project:

    # At the root directory of your project
    git submodule add https://github.com/MotohiroOGAWA/MassMolKit.git cores/MassMolKit
    git commit -m "Add MassMolKit as submodule"

## 🔄 Updating Submodules

    cd cores/MassMolKit
    git checkout main
    git pull origin main
    cd ../..
    git add cores/MassMolKit
    git commit -m "Update MassMolKit submodule"

## ⚙️ Installation (Editable Mode)

To install MassMolKit into your Python environment for development, run:

    cd cores/MassMolKit
    pip install -e .

The -e option (editable mode) ensures that any modifications to the source code are immediately reflected in your environment.

---

## 🧪 Running Tests

You can run tests to verify that MassMolKit is working properly.

### Using Python

    python -m MassMolKit.run_tests

### Using the shell script

    ./MassMolKit/run_tests.sh


