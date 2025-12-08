from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ree_visualization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # CORRECTION : Copier le layout depuis le bon emplacement
        ('share/' + package_name, ['ree_visualization/foxglove_layout.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sojavola',
    maintainer_email='sojavolar.2002@gmail.com',
    description='Foxglove visualization for REE exploration system',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
             'foxglove_viz_node = ree_visualization.foxglove_visualization_node:main',
        ],
    },
)