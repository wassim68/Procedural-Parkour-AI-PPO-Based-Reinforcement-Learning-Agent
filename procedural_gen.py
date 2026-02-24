import random
from platforms import Platform
import consts


random.seed(42)


MIN_GAP = 100       
MAX_GAP = 300         
MIN_HEIGHT = 200        
MAX_HEIGHT = 500        
PLATFORM_WIDTH_RANGE = (80, 150)  
PLATFORM_HEIGHT = 20

MIDDLE = consts.SCREEN_WIDTH/2

def generate_next_platform(prev_platform):
    
    # horizontal gap
    gap = random.randint(MIN_GAP, MAX_GAP)
    next_x = prev_platform.pos.x + prev_platform.rect.width + gap

    # vertical height
    min_y = int(max(MIN_HEIGHT, prev_platform.pos.y - 100))
    max_y = int(min(MAX_HEIGHT, prev_platform.pos.y + 100))
    next_y = random.randint(min_y, max_y)

    # width
    width = random.randint(*PLATFORM_WIDTH_RANGE)

    return Platform(next_x, next_y, width, PLATFORM_HEIGHT)


def generate_initial_platforms(start_x = MIDDLE , num_platforms=5):
    seed = random.randint(0, 10000)
    random.seed(seed)
    platforms = []
    x = start_x
    y = consts.SCREEN_HEIGHT - 150    
    for _ in range(num_platforms):
        width = random.randint(*PLATFORM_WIDTH_RANGE)
        platforms.append(Platform(x, y, width, PLATFORM_HEIGHT))
        gap = random.randint(MIN_GAP, MAX_GAP)
        x += width + gap
        y += random.randint(-50, 50)  
        y = max(MIN_HEIGHT, min(MAX_HEIGHT, y))
    return platforms