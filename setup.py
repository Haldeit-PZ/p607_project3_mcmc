# Always prefer setuptools over distutils

from setuptools import setup, find_packages

setup(
    name='box2D',
    version='1.0.1',
    author='Luis Rufino, Haoyang Zhou',
    description="A Markov Chain, Monte-Carlo Simulation of 1D Harmonic Oscillator in Quantum Field Theory", 
    author_email='sopherium@protonmail.com, luisrufino24@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Haldeit-PZ/p607_project3_mcmc.git', 
    package_dir={"": "src",}, 
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'imageio',
        'natsort'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'my-command=scripts.my_script:main',
    #     ],
    # },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    scripts=['bin/simulateOscillator'],
)