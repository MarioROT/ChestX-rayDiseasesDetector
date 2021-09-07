import setuptools

setuptools.setup(
    name="ChestX-ray_Disease_Detector",
    version="0.0.1",
    author="Mario Rosas O.",
    author_email="mario_netta12@hotmail.com",
    url="https://github.com/Mariuki/ChestX-rayDiseasesDetector",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scikit-image",
        "sklearn",
        "importlib_metadata",
        "neptune-contrib",
        "napari[all]==0.4.9",
        "jupyterlab==3.0.13",
        "ipywidgets==7.6.3",
        "albumentations==0.5.2",
        "pytorch-lightning==1.3.5",
        "magicgui==0.2.9",
        "torch==1.8.1",
        "torchvision==0.9.1",
        "torchsummary==1.5.1",
        "torchmetrics==0.2.0",
    ],
)
