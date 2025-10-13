from setuptools import find_packages, setup

package_name = 'ree_exploration_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
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
            'agent_node = ree_exploration_agent.agent_node:main',
            'multi_agent_coordinator = ree_exploration_agent.multi_agent_coordinator:main',
#            'dqn_trainer = ree_exploration_agent.advanced_dqn_agent:main',
        ],
    },
)
