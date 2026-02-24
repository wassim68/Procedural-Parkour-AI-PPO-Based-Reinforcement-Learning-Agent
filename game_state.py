import platforms
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from camera import Camera
import platforms
import player
import procedural_gen
import consts


class GameState(gym.Env):       
    delta_time = 1.0 / 165.0
    last_x =  consts.PLAYER_START_POSITION[0]
    max_x_reached = consts.PLAYER_START_POSITION[0]
    last_max_x_update_time = 0.0
    total_reward = 0.0
    steps = 0
    _action_timers = {"right": 0.0,"left": 0.0, "Jump": 0.0, "Dash": 0.0}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(5)  
        #[0 nothing, 1 left, 2 right, 3 jump, 4 dash]

        self.observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(20,),
        dtype=np.float32
        )

        # Observation vector (all values normalized to [-1, 1]):
        # [py, vx, vy,
        #  on_ground, can_double_jump, dash_available,is_dashing,dash_cooldown,
        #  pp_dx, pp_dy, pp_w,
        #  p1_dx, p1_dy, p1_w,
        #  p2_dx, p2_dy, p2_w,
        #  p3_dx, p3_dy, p3_w]

        self.FPS = 165
        self.delta_time = 1.0 / self.FPS

        self._build_world()

    def _build_world(self):
        self.platforms = procedural_gen.generate_initial_platforms(
            start_x=consts.PLAYER_START_POSITION[0],
            num_platforms=8
        )

        first_platform = self.platforms[0]
        y_start = first_platform.rect.top - player.PLAYER_HEIGHT
        self.player = player.Player(
            consts.PLAYER_START_POSITION[0],
            y_start
        )


        self.total_progress = 0
        self.steps = 0
        self.total_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_world()
        self.max_x_reached = self.player.pos.x
        self.last_x = self.player.pos.x
        self.total_reward = 0.0
        self.steps = 0
        return self._get_state(), {}
    

    def step(self,action):

        self.steps += 1

        # translate discrete action into movement + action

        self.player.current_move.x = 0

        if action == 1:
            self.player.current_move.x = -1
        elif action == 2:
            self.player.current_move.x = 1
     
        player_action = 0
        if action == 3:
            player_action = 1  # jump
        elif action == 4:
            player_action = 2  # dash
        

        # update world
        platform_reached = self.player.update(player_action, self.delta_time, self.platforms)

        # generate new platforms
        if hasattr(self.player, "progress"):
            if self.player.progress >= len(self.platforms) - 6 or self.player.pos.x > self.platforms[-1].pos.x - 200:
                last_platform = self.platforms[-1]
                for _ in range(5):
                    new_platform = procedural_gen.generate_next_platform(last_platform)
                    self.platforms.append(new_platform)
                    last_platform = new_platform

        #reward calculation     

        reward = 0.0
        terminated = False
        truncated = False

        # forward progress  
        progress_delta = self.player.pos.x - self.last_x

        if progress_delta > 0:
            reward += 0.005 * progress_delta

        self.last_x = self.player.pos.x


        # new max position bonus 

        if self.player.pos.x > self.max_x_reached:
            reward += 0.2
            self.max_x_reached = self.player.pos.x
            self.last_max_x_update_time = 0.0
        else:
            self.last_max_x_update_time += self.delta_time



        # new Platform reward 
        if platform_reached > 0:
            reward += 3.0 * platform_reached


        # time penalty
        reward -= 0.002

        # stalling penalty
        if self.last_max_x_update_time > 0.5:
            reward -= 1.0 
            

        # fall penalty

        if self.player.pos.y + self.player.rect.height >= consts.SCREEN_HEIGHT:
            reward -= 30.0
            terminated = True


        # time limit
        if self.steps > 10000:
            truncated = True


        # clip reward
        if not terminated:
            reward = max(-5.0, min(5.0, reward))


        self.total_reward += reward 

        platforms.clearColors(self.platforms)

        return self._get_state(), reward, terminated, truncated, {}     

    def _get_state(self):

        # sort platforms by x
        platforms_sorted = sorted(self.platforms, key=lambda p: p.pos.x)

        # next 3 platforms ahead of player
        upcoming = [p for p in platforms_sorted if p.pos.x > self.player.pos.x]
        previous = [p for p in platforms_sorted if p.pos.x <= self.player.pos.x]
        last_y = platforms_sorted[-1].pos.y

        # previous platform  
        pp = previous[-1] if previous else platforms.Platform(
            x=self.player.pos.x - consts.SCREEN_WIDTH,  
            y=last_y,  
            width=0,  
            height=0
        )
        if len(upcoming) < 3:
            while len(upcoming) < 3:
                dummy = platforms.Platform(
                    x=self.player.pos.x + consts.SCREEN_WIDTH,  
                    y=last_y,                                  
                    width=0,                                   
                    height=0
                )
                upcoming.append(dummy)

        p1, p2, p3 = upcoming[0], upcoming[1], upcoming[2]

        pp.color = (255, 0, 255)  # magenta for previous platform
        p1.color = (0, 100, 0)     # green for next platform
        p2.color = (0, 175, 0)   # yellow for second next platform
        p3.color = (0, 255, 0)   # orange for third next platform

        # normalize values
        py = self.player.pos.y / consts.SCREEN_HEIGHT
        vx = self.player.vel.x / player.MAX_VEL_X
        vy = self.player.vel.y / player.MAX_VEL_Y

        dxpp = (pp.pos.x - self.player.pos.x) / consts.SCREEN_WIDTH
        dxpp = max(-1.0, min(1.0, dxpp))
        dypp = (pp.pos.y - self.player.pos.y) / consts.SCREEN_HEIGHT
        wpp = pp.rect.width / consts.SCREEN_WIDTH

        dx1 = (p1.pos.x - self.player.pos.x) / consts.SCREEN_WIDTH
        dx1 = max(-1.0, min(1.0, dx1))
        dy1 = (p1.pos.y - self.player.pos.y) / consts.SCREEN_HEIGHT
        w1 = p1.rect.width / consts.SCREEN_WIDTH

        dx2 = (p2.pos.x - self.player.pos.x) / consts.SCREEN_WIDTH
        dx2 = max(-1.0, min(1.0, dx2))
        dy2 = (p2.pos.y - self.player.pos.y) / consts.SCREEN_HEIGHT
        w2 = p2.rect.width / consts.SCREEN_WIDTH

        dx3 = (p3.pos.x - self.player.pos.x) / consts.SCREEN_WIDTH
        dx3 = max(-1.0, min(1.0, dx3))
        dy3 = (p3.pos.y - self.player.pos.y) / consts.SCREEN_HEIGHT
        w3 = p3.rect.width / consts.SCREEN_WIDTH

        # other player stats
        on_ground = 1.0 if self.player.on_ground else -1.0
        can_double_jump = 1.0 if self.player.can_jump() else -1.0
        dash_available = 1.0 if not self.player.is_dashing  and self.player.dash_delay <= 0 else -1.0
        is_dashing = 1.0 if self.player.is_dashing else -1.0
        dash_cooldown = self.player.dash_delay / player.DASH_DELAY if self.player.dash_delay > 0 else -1.0


        #state
        state = np.array(
            [py, vx, vy,
            on_ground, can_double_jump,
            dash_available, is_dashing, dash_cooldown,
            dxpp, dypp, wpp,
            dx1, dy1, w1,
            dx2, dy2, w2,
            dx3, dy3, w3
            ],
            dtype=np.float32
        )

        return state

    def init(self,RENDER=False):
        
        if RENDER:

            self.screen = pygame.display.set_mode((consts.SCREEN_WIDTH, consts.SCREEN_HEIGHT))
            self.camera = Camera(consts.SCREEN_WIDTH, consts.SCREEN_HEIGHT)      
        self.player = player.Player(
            consts.PLAYER_START_POSITION[0],
            consts.PLAYER_START_POSITION[1] - platforms.DEFAULT_PLATFORM_HEIGHT - player.PLAYER_HEIGHT/2
        )
        self.platforms = []     

    def render(self, total_reward=0.0, recent_reward=0.0, current_action=None, best_platform_index=0):
        self.screen.fill((0, 100, 150))

        # draw floor
        floor_rect = pygame.Rect(0, consts.SCREEN_HEIGHT, consts.SCREEN_WIDTH, 50)
        draw_pos_y = floor_rect.topleft[1] - self.camera.pos.y
        pygame.draw.rect(self.screen, (255, 255, 255), (floor_rect.topleft[0], draw_pos_y, floor_rect.width, floor_rect.height))

        # draw platforms
        for platform in self.platforms:
            draw_pos = platform.pos - self.camera.pos
            pygame.draw.rect(self.screen, platform.color, (draw_pos.x, draw_pos.y, platform.rect.width, platform.rect.height))

        # draw player
        self.player.draw(self.screen, self.camera)

        # font for text
        font = pygame.font.SysFont("Arial", 18)

        # print total score
        text_total = font.render(f"Total Score: {total_reward:.2f}", True, (255, 255, 0))
        self.screen.blit(text_total, (10, 10))

        # print recent reward rate 0.3 second 
        text_recent = font.render(f"Recent Reward/sec: {recent_reward:.2f}", True, (255, 200, 0))
        self.screen.blit(text_recent, (10, 30))

        # print best platform reached
        text_platform = font.render(f"Best Platform Index: {best_platform_index}", True, (0, 255, 0))
        self.screen.blit(text_platform, (10, 50))

        current_action_int = current_action

        
        # horizontal (Left/Right)
        horizontal_str = ""

        if (current_action_int == 1):  # Left
            self._action_timers["left"] = 0.33
            horizontal_str = "  ← "
        elif self._action_timers["left"] > 0:
            horizontal_str = "  ← "  
        else:
            horizontal_str = "None"

        if (current_action_int == 2):  # Right
            self._action_timers["right"] = 0.33
            horizontal_str += " /  →  "
        elif self._action_timers["right"] > 0:
            horizontal_str += " /  →  "  
        else:
            horizontal_str += " / None"  

        # jump
        if current_action_int == 3:  # Jump 
            self._action_timers["Jump"] = 0.33
            jump_str = "↑ Jump"
        elif self._action_timers["Jump"] > 0:
            jump_str = "↑ Jump"
        else:
            jump_str = "  None"

        # dash
        if current_action_int == 4:  # Jump 
            self._action_timers["Dash"] = 0.33
            dash_str = "---> dash"
        elif self._action_timers["Dash"] > 0:
            dash_str = "---> dash"
        else:
            dash_str = "     None"

        # decrease timers based on delta_time
        dt = self.delta_time
        for k in self._action_timers:
            self._action_timers[k] = max(0.0, self._action_timers[k] - dt)

        #draw action text
        color_on = (255, 0, 255)
        color_off = (0, 0, 0)
        text_horizontal = font.render(f"Horizontal: {horizontal_str}", True, color_on if current_action_int in [1,2] else color_off)
        text_jump = font.render(f"Jump: {jump_str}", True, color_on if current_action_int == 3 else color_off)
        text_dash = font.render(f"Dash: {dash_str}", True, color_on if current_action_int == 4 else color_off)

        self.screen.blit(text_horizontal, (10, 70))
        self.screen.blit(text_jump, (10, 90))
        self.screen.blit(text_dash, (10, 110))

        # flip display
        pygame.display.flip()