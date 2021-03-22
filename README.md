# pip_tutorial
This is a simple example package. 


https://zhuanlan.zhihu.com/p/342682533


### How to build your pip wheel
#### Part 1
1.1 Sign up for an account

1.2 Code Package Management
        
        pip_tutorial
        └── laneSegFcn
            └── __init__.py

1.3 Create package files

    A total of LICENSE, README, setup.py, and if you want to test, you can use the tests folder
    
    pip_tutorial
    ├── LICENSE
    ├── README.md
    ├── laneSegFcn
    │   └── __init__.py
    ├── setup.py
    └── tests

1.4 Create setup.py

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

#### Part 2
2 Create Release Archive

2.1 Install the latest setuptools and wheel

2.2 Mkdir dist, build

2.3 Run in the directory of setup.py

    python3 setup.py sdist bdist_wheel
    
This will find the package tar.gz file (which is the source archive) and the whl file (which is the built distribution) in the dist directory.

The latest pip versions will prefer to install the whl file, but will also trace back to the source archive if necessary.

#### Part 3 
3 Upload Release Archive

Method 1: 
    
    twine upload dist/*

not finished

