from time import sleep
import platforms as plat
import pygame
import player as playerF
import procedural_gen
import consts
import camera




pygame.init()


FPS = 165
clock = pygame.time.Clock()
dl = FPS / 1000
running = True

screen = pygame.display.set_mode((consts.SCREEN_WIDTH, consts.SCREEN_HEIGHT))

player = playerF.Player(consts.PLAYER_START_POSITION[0], consts.PLAYER_START_POSITION[1] - plat.DEFAULT_PLATFORM_HEIGHT - playerF.PLAYER_HEIGHT/2)


platforms = procedural_gen.generate_initial_platforms(start_x=consts.PLAYER_START_POSITION[0], num_platforms= 8)

camera = camera.Camera(consts.SCREEN_WIDTH, consts.SCREEN_HEIGHT)

action = 0

def eventHandler(player,):
    global action, running
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # get keys
    keys = pygame.key.get_pressed()

    # horizontal
    player.current_move.x = 0
    if keys[pygame.K_d]:
        player.current_move.x = 1
    elif keys[pygame.K_a]:
        player.current_move.x = -1


    # actions

    if keys[pygame.K_SPACE] or keys[pygame.K_w] :
        action = 1  # jump
    elif keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        action = 2  # dash
    else: action = 0

    #reset 

    if keys[pygame.K_r]:
        player.reset(consts.PLAYER_START_POSITION[0], consts.PLAYER_START_POSITION[1] - plat.DEFAULT_PLATFORM_HEIGHT - playerF.PLAYER_HEIGHT/2)
        camera.pos = pygame.Vector2(0, 0)

def updatePlatforms(dl):
    for platform in platforms:
        platform.update(dl)

def drawPlatforms(screen):
    for platform in platforms:
        platform.draw(screen, camera)

def checkProgress():
    global platforms, player
    if player.progress >= len(platforms) - 6:
        last_platform = platforms[-1]
        for _ in range(5):
            new_platform = procedural_gen.generate_next_platform(last_platform)
            platforms.append(new_platform)
            last_platform = new_platform
             
    
def update(dl):
    global action
     
    player.update(action, dl,platforms)   
    updatePlatforms(dl)
    camera.follow(player.pos, dl)
    checkProgress()

def drawFloor():
    floor_rect = pygame.Rect(0, consts.SCREEN_HEIGHT , consts.SCREEN_WIDTH, 50)
    draw_pos_y = floor_rect.topleft[1] - camera.pos.y
    pygame.draw.rect(screen, (255, 255, 255), (floor_rect.topleft[0], draw_pos_y, floor_rect.width, floor_rect.height))

def draw():
    screen.fill((0,100,150)) 
    drawFloor()
    player.draw(screen, camera)
    drawPlatforms(screen)
    pygame.display.flip()



while running:
    
    eventHandler(player)

    update(dl)

    draw()


    dl = clock.tick(FPS) / 1000

    