from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'TF modeling with Deep RegulAtory GenOmic Neural Networks (TF-DragoNN)',
    'download_url': 'https://github.com/kundajelab/tf-dragonn',
    'version': '0.1.0',
    'packages': ['tfdragonn'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras==0.3.2', 'deeplift', 'sklearn', 'future', 'psutil'],
    'dependency_links': ['https://github.com/kundajelab/deeplift/tarball/master#egg=deeplift-0.2'],
    'scripts': ["scripts/test_io_utils.py"],
    'entry_points': {'console_scripts': ['tfdragonn = tfdragonn.__main__:main']},
    'name': 'tfdragonn'
}

if __name__== '__main__':
    setup(**config)
