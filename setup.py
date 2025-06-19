from setuptools import setup

setup(
    name="acaload",
    version="0.1",
    py_modules=["model"],
    install_requires=[
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "yt-dlp>=2024.1.1",
    ],
    entry_points={
        "console_scripts": [
            "acaload=model:main",  # assumes your script has a `main()` entry
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
