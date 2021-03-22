import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DuckDuckZz", # Replace with your own username
    version="0.0.1",
    author="Zeqi",
    author_email="author@example.com",
    description="A small example package 2021322",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monchhichizzq/pip_tutorial",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=['laneSegFcn.bb']
)
