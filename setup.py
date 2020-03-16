from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read();

def license():
    with open('LICENSE') as f:
        return f.read();

setup(
    name='twitter-language-identification',
    version='0.0.1',
    description='A language identification system focusing on Twitter posts',
    long_description=readme(),
    url='https://github.com/eyeonechi/twitter-language-identification',
    author='Ivan Ken Weng Chee',
    author_email='ichee@student.unimelb.edu.au',
    license=license(),
    keywords=[],
    scripts=[
        'src/twitter_language_identification.py'
    ],
    packages=[],
    zip_safe=False,
    include_package_data=True
)
