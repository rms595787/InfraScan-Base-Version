# 🛰️ Project Setup and Requirements

> A reliable and optimized environment for accurate image processing and comparison.

## 🎛️ Table of Contents

1. [Python Version Requirement](#-python-version-requirement)
2. [Required Libraries](#-required-libraries)
3. [Why These Versions](#-why-these-versions)
4. [Running the Project](#-running-the-project)
5. [Output Preview](#-output-preview)
6. [Conclusion](#-conclusion)

[![Python Version](https://img.shields.io/badge/Python-3.11%20to%203.12-blue?logo=python)](https://www.python.org/downloads/release/python-3110/)
[![Status](https://img.shields.io/badge/Project-Stable-brightgreen)](#)
[![Platform](https://img.shields.io/badge/Platform-Cross--OS-lightgrey?logo=windows&logoColor=white)](https://en.wikipedia.org/wiki/Cross-platform_software)
[![Framework](https://img.shields.io/badge/Uses-Flask-orange?logo=flask)](https://flask.palletsprojects.com/en/stable/)

This project requires specific versions of Python and certain libraries to function correctly. Please follow the installation and usage instructions carefully to avoid compatibility issues.

## ✅ Python Version Requirement

Use **Python version between 3.11 and 3.12**.

> ⚠️ Note: Using Python versions **above 3.12** may cause errors because **some library syntaxes and internal dependencies have changed** in the newer versions. This may lead to unexpected results or the project not running correctly.

## 📦 Required Libraries

Install all required dependencies using the following command:

```
pip install numpy==1.24.4 opencv-python==4.8.1.78 scikit-image==0.21.0 matplotlib flask flask-cors
```

## 🛠️ Why These Versions?

- The project uses functions and interfaces from these libraries that are **tested and compatible** with Python 3.11–3.12.
- Newer versions of Python introduce changes in C-API and underlying modules used by `numpy`, `opencv-python`, and `scikit-image`, which may lead to build or runtime errors.

## 🚀 Running the Project

1. Ensure you are using Python 3.11 or 3.12
2. Install dependencies using the above pip command
3. Run your main script:

```
python index.py
```

## 🎯 Conclusion

Stick to the recommended Python version range to ensure:

- Smooth execution
- Correct processing of images and calculations
- Proper compatibility with the installed library versions

If you face any setup issues, verify Python version by running:

```
python --version
```

---

## 📹 Output Preview

[![Watch on LinkedIn](https://img.shields.io/badge/Watch%20Demo-LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/feed/update/urn:li:activity:7392302232068612097/?originTrackingId=UFYawNcWRwS%2FSq1P6%2F60bA%3D%3D)

Happy Coding! 🚀
