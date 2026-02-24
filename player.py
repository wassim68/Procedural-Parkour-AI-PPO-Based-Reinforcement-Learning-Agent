import pygame
from pygame.math import Vector2

import consts



PlAYER_WIDTH = 50
PLAYER_HEIGHT = 50

JUMP_POWER = 700
COYOTE_TIME = 0.1
SECOND_JUMP_DELAY = 0.2

DASH_DISTANCE = 300
DASH_VELOCITY = 3000
DASH_DELAY = 1500
DASH_ENABLED = True

GRAVITY = 1500

AIR_DRAG_Y = 0.1
AIR_DRAG_X = 5

GROUND_FRICTION = 30

VELOCITY_THRESHOLD = 25

MAX_VEL_X = 800
MAX_VEL_Y = 2000

ACCELERATION = 8000

NEW_PLATFORM_REWARD = 2
OLD_PLATFORM_PENALTY = 0
GAP_REWARD_THRESHOLD = 250

class Player:
    rect = pygame.Rect(0,0, PlAYER_WIDTH, PLAYER_HEIGHT)
    pos = Vector2(0,0)
    vel = Vector2(0,0)
    on_ground = False
    is_dashing = False
    dash_direction = 1
    dashing_distance = 0
    dash_delay = 0
    current_move = Vector2(0,0)
    progress = 0
    coyote_timer = 0
    second_jump_available = True
    second_jump_delay = 0


    def can_jump(self):
        return self.on_ground or self.coyote_timer > 0 or (self.second_jump_available and self.second_jump_delay <= 0)
    

    def updateRect(self):
        self.rect.x = self.pos.x
        self.rect.y = self.pos.y

    def applyDrag(self, delta_time):
        if  not self.is_dashing and self.current_move.x == 0:
            if abs(self.vel.x) <= VELOCITY_THRESHOLD:
                self.vel.x = 0
            else:
                self.vel.x -= self.vel.x * AIR_DRAG_X * delta_time

        
        self.vel.y -= self.vel.y * AIR_DRAG_Y * delta_time

    def applyGravity(self,delta_time):
        if not self.on_ground and not self.is_dashing :
            self.vel.y += GRAVITY  * delta_time

    def applyVelocity(self, delta_time):
        
            if not self.on_ground:
                self.pos.y += self.vel.y * delta_time
            self.pos.x += self.vel.x * delta_time

    def clampVelocity(self):
        self.vel.x = max(-MAX_VEL_X, min(MAX_VEL_X, self.vel.x))
        self.vel.y = max(-MAX_VEL_Y, min(MAX_VEL_Y, self.vel.y))    

    def applyFriction(self, delta_time):
        if self.on_ground and not self.is_dashing and self.current_move.x == 0:
                self.vel.x -= self.vel.x * GROUND_FRICTION * delta_time

        if abs(self.vel.x) <= VELOCITY_THRESHOLD:
                self.vel.x = 0

    def addMovement(self,delta_time):
        self.pos.x += self.current_move.x * MAX_VEL_X * delta_time
        

    def applyAction(self,action):
        #action 0 -> do nothing
        #action 1 -> jump
        if action == 1 : 
            if self.on_ground or self.coyote_timer > 0:
                self.vel.y = -JUMP_POWER
                self.on_ground = False
                self.coyote_timer = 0
            elif  self.second_jump_available and self.second_jump_delay <= 0:
                self.vel.y = -JUMP_POWER
                self.second_jump_available = False    
        #action 2 -> dash    
        if action == 2 and not self.is_dashing  and self.dash_delay <= 0 and DASH_ENABLED :
            self.is_dashing = True
            self.dash_distance = DASH_DISTANCE
            self.dash_delay = DASH_DELAY
            

    def applyDash(self, delta_time):
        if self.is_dashing:
            dash_fraction = min(self.dash_distance / DASH_DISTANCE , 0.9)   

            multiplier = (1 - dash_fraction**3)   

            delta_x = DASH_VELOCITY * delta_time * self.dash_direction * multiplier

            if abs(delta_x) <= self.dash_distance:
                self.pos.x += delta_x
                self.dash_distance -= abs(delta_x)
            else:
                self.pos.x += self.dash_distance * self.dash_direction
                self.is_dashing = False

    def update(self, action , delta_time,platforms = [] ):
        
        self.clampVelocity()
        self.addMovement(delta_time)  
        self.applyAction(action)
        self.applyGravity(delta_time)
        self.applyVelocity(delta_time)
        self.applyDrag(delta_time)
        self.applyFriction(delta_time)
        self.applyDash(delta_time)
        self.restore_jump()
        reward = self.check_collisions(platforms)
        
        # dash cooldown
        self.dash_delay -= delta_time*1000
        if self.dash_delay < 0:
            self.dash_delay = 0

        # update coyote timer
        self.coyote_timer -= delta_time
        if self.coyote_timer < 0:
            self.coyote_timer = 0

        self.second_jump_delay -= delta_time
        if self.second_jump_delay < 0:
            self.second_jump_delay = 0    


        #finnaly
        self.updateRect()

        return reward

    def restore_jump(self):
        if self.on_ground:
            self.second_jump_available = True
            self.second_jump_delay = SECOND_JUMP_DELAY
    
    def check_collisions(self, platforms):
        before_on_ground = self.on_ground
        # side collisions

        if self.pos.x < 0:
            self.pos.x = 0
            self.vel.x = 0
            self.is_dashing = False
            self.dash_distance = 0

    
        self.on_ground = False

        # floor collision
        if self.pos.y + self.rect.height >= consts.SCREEN_HEIGHT:
            self.pos.y = consts.SCREEN_HEIGHT - self.rect.height
            self.vel.y = 0
            self.on_ground = True
            
            return 0

        #  collisions 
        if self.vel.y >= 0:  
            for platform in platforms:
                if self.rect.colliderect(platform.rect):
                    if self.pos.y + self.rect.height - self.vel.y * 0.05 <= platform.rect.top:
                        self.pos.y = platform.rect.top - self.rect.height 
                        self.vel.y = 0
                        self.on_ground = True
                        index = platforms.index(platform)
                        if index > self.progress:
                            progress = abs(index - self.progress)
                            reward = NEW_PLATFORM_REWARD * progress 
                            gap = platform.rect.x - (platforms[self.progress].rect.x + platforms[self.progress].rect.width)
                            if gap > GAP_REWARD_THRESHOLD:
                                reward += 2.0 * progress
                            self.progress = index
                            return reward
                        elif index == self.progress:
                            return 0
                        else:
                            progress = abs(index - self.progress)
                            reward = OLD_PLATFORM_PENALTY * progress
                            return reward
                        break
        if not before_on_ground and self.on_ground:
            self.coyote_timer = COYOTE_TIME  
        return 0              
        
        
    def __init__(self, x, y):
        self.pos = Vector2(x,y)
        self.updateRect()
        self.vel = Vector2(400,0)
        self.on_ground = False
        self.is_dashing = False
        self.progress = 0

    def reset(self, x, y):
        self.pos = Vector2(x, y)
        self.vel = Vector2(0, 0)
        self.on_ground = False
        self.is_dashing = False
        self.dash_distance = 0
        self.updateRect()
        self.progress = 0

    def draw(self, screen, camera):
        draw_pos = self.pos - camera.pos
        pygame.draw.rect(screen, (255,0,0), (draw_pos.x, draw_pos.y, self.rect.width, self.rect.height))

    def get_state(self, platforms):
        nearest = min(platforms, key=lambda p: p.rect.x - self.pos.x if p.rect.x > self.pos.x else float('inf'))
        state = [
            self.pos.x,
            self.pos.y,
            self.vel.x,
            self.vel.y,
            nearest.rect.x - self.pos.x,
            nearest.rect.y - self.pos.y,
        ]
        return state