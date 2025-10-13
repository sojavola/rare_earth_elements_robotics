import rclpy
import sys
from rclpy.node import Node
from .advanced_dqn_agent import AdvancedDQNAgent
from .science_reward_system import ScienceRewardSystem

class AgentNode(Node):
    def __init__(self, robot_id=0):
        super().__init__(f'agent_node_{robot_id}')
        self.robot_id = robot_id
        
        # Initialiser l'agent DQN et le système de récompenses
        state_shape = (100, 172, 4)  # À adapter selon vos besoins
        self.dqn_agent = AdvancedDQNAgent(state_shape, num_actions=8)
        self.reward_system = ScienceRewardSystem()
        
        self.get_logger().info(f'Agent Node {robot_id} initialized')

def main():
    rclpy.init()
    
    # Récupérer l'ID du robot depuis les arguments
    robot_id = 0
    if len(sys.argv) > 1:
        robot_id = int(sys.argv[1])
    
    node = AgentNode(robot_id)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()