import rclpy
from rclpy.node import Node

class WorldManager(Node):
    def __init__(self):
        super().__init__('world_manager')
        self.get_logger().info('🌕 World Manager started - Lunar REE Exploration Ready!')
        
        # Timer pour le monitoring
        self.create_timer(5.0, self.monitor_status)

    def monitor_status(self):
        """Surveille l'état du monde"""
        self.get_logger().info('🛰️  World status: ACTIVE - REE exploration environment operational')

def main():
    rclpy.init()
    node = WorldManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()