from setuptools import setup, find_packages
setup(
    name='arcontrol2nwb',
    version='0.1', 
    packages=find_packages(include=['arcontrol2nwb*']),
    url='https://github.com/chenxinfeng4/ArControl-covert2-nwb',
    author='chenxinfeng',
    install_requires=['pynwb','numpy']
)
