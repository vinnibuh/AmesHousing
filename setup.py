from setuptools import setup

NAME = "AmesHousing"

DESCRIPTION = "SE in ML course project at MIPT"

AUTHOR = "Ivchenkov Yaroslav"

AUTHOR_EMAIL = "ivchenkov.yap@phystech.edu"

URL = "https://github.com/vinnibuh/AmesHousing"

VERSION = "1.0.3"

install_requires = []
with open(f"./requirements.txt", 'r') as f:
    for line in f.readlines():
        line = line.split('#')[0].strip()
        if len(line) > 0:
            install_requires.append(line)

tests_require = [
    'pytest',
    'mutmut'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=install_requires,
    tests_require=tests_require,
    package_dir={'': 'src'},
    packages=['housinglib', ],
)
