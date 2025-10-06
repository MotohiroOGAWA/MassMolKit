# MassMolKit

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![RDKit](https://img.shields.io/badge/RDKit-2024.03.5-green)

---

MassMolKit is a Python toolkit for handling molecules and mass spectrometry data.  
It provides utilities for canonical SMILES conversion, formula handling, and MS-related operations.  
It can be integrated as a Git submodule or installed in editable mode.

---

## 🔗 Add as a Git Submodule

To integrate MassMolKit into an existing project, run the following commands at the root of your project:

```bash
# At the root directory of your project
git submodule add https://github.com/MotohiroOGAWA/MassMolKit.git ./MassMolKit
git commit -m "Add MassMolKit as submodule"
```

## ⚙️ Install from GitHub (Recommended for Users)
You can install **MassMolKit** directly from the GitHub repository:
```bash
pip install git+https://github.com/MotohiroOGAWA/MassMolKit.git
```

## 🧪 Running Tests
You can run tests to verify that MassMolKit is working properly.

### Using Python
```bash
python -m run_tests
```

### Using the shell script
```bash
./run_tests.sh
```