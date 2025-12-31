from setuptools import setup, find_packages

def get_requirements(requirements_path: str) -> list[str]:
    """Read the requirements from a file and return them as a list."""
    with open(requirements_path, 'r') as file:
        requirements = [line.strip() for line in file if line.strip() and not line.startswith('#')]
        requirements = [req for req in requirements if req != "-e ."]

    return requirements

setup(
    name='human-activity-recognition',
    version='0.1.0',
    author='Samsudeen Lawal',
    packages=find_packages(),
    install_requires=get_requirements(requirements_path='requirements.txt'),
    entry_points={
        'console_scripts': [
            'human_activity_recognition=human_activity_recognition.main:main',
        ],
    },
    description='A package for human activity recognition using machine learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)