import pygame


class Camera:
    def __init__(self, width, height):
        self.pos = pygame.Vector2(0, 0)
        self.width = width
        self.height = height

    def follow(self, target, delta_time, smooth=5):
        
        
        target_x = target.x - self.width // 2
        target_y = target.y - self.height // 2

        
        self.pos.x += (target_x - self.pos.x) * smooth * delta_time
        self.pos.y += (target_y - self.pos.y) * smooth * delta_time