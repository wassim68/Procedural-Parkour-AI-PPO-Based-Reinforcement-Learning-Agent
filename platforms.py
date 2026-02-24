import pygame

DEFAULT_PLATFORM_WIDTH = 100
DEFAULT_PLATFORM_HEIGHT = 20
DEFAULT_PLATFORM_COLOR = (255, 255, 255)



class Platform:
    rect = pygame.Rect(0,0, DEFAULT_PLATFORM_WIDTH, DEFAULT_PLATFORM_HEIGHT)
    def __init__(self, x, y, width=DEFAULT_PLATFORM_WIDTH, height=DEFAULT_PLATFORM_HEIGHT, color=DEFAULT_PLATFORM_COLOR):
        self.pos = pygame.Vector2(x, y)
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
    
    def update(self, delta_time):
        pass
    
    def move(self, dx, dy):
        self.pos.x += dx
        self.pos.y += dy
        self.update_rect()
    
    def update_rect(self):
        self.rect.topleft = (self.pos.x, self.pos.y)
    
    def draw(self, screen, camera, color=None):
        draw_pos = self.pos - camera.pos
        pygame.draw.rect(screen, color if color else self.color, (draw_pos.x, draw_pos.y, self.rect.width, self.rect.height))
    
    def get_state(self, player_pos=None):
        if player_pos:
            return [self.pos.x - player_pos.x, self.pos.y - player_pos.y,
                    self.rect.width, self.rect.height]
        else:
            return [self.pos.x, self.pos.y, self.rect.width, self.rect.height]
        
def clearColors(platforms):
    for platform in platforms:
        platform.color = (0, 0, 0)        