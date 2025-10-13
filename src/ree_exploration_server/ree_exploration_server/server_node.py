import rclpy
from rclpy.node import Node
from .advanced_mineral_generator import AdvancedMineralGenerator

class ServerNode(Node):
    def __init__(self):
        super().__init__('server_node')
        
        # Initialiser le générateur de carte minérale
        self.mineral_generator = AdvancedMineralGenerator(width=172, height=100)
        
        self.get_logger().info('REE Exploration Server Node initialized')

def main():
    rclpy.init()
    node = ServerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()