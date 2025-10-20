from setuptools import find_packages, setup

package_name = 'generative_ai_layer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools'
        'rclpy',
        'std_msgs',
        'geometry_msgs',
        'sensor_msgs',
        # Versions compatibles Pydantic v2
        'pydantic>=2.0.0',
        'pydantic-core>=2.0.0',
        'langchain-core>=0.3.0',
        'langchain-community>=0.3.0',
        'langchain>=0.3.0',
        'langchain-mistralai>=0.1.0',
        'mistralai>=0.1.0',
        'transformers>=4.35.0',
        'torch>=2.1.0',
        'numpy>=1.24.3',
        'requests>=2.31.0',
        ],
    zip_safe=True,
    maintainer='sojavola',
    maintainer_email='sojavolar.2002@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
#        'test': [
#            'pytest',
#        ],

    },
    entry_points={
        'console_scripts': [
             'science_ai_system = generative_ai_layer.science_ai_system:main',
             ],
    },
)

