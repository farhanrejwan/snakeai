import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum) :
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x,y')

WHITE = (250, 250, 250)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
GAME_SPEED = 2000

class SnakeGameAI :
    def __init__(self, w=640, h=480) :
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self) :
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
    
    def place_food(self) :
        x = random.randint(0, (self.w - BLOCK_SIZE ) // BLOCK_SIZE ) * BLOCK_SIZE 
        y = random.randint(0, (self.h - BLOCK_SIZE ) // BLOCK_SIZE ) * BLOCK_SIZE
        
        self.food = Point(x,y)
        
        if self.food in self.snake :
            self.place_food()
        
    def play_step(self, action) :
        self.frame_iteration += 1
        
        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                pygame.quit()
                quit()
        
        self.move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake) :
            game_over = True
            reward = -10
            
            return reward, game_over, self.score
            
        if self.head == self.food :
            self.score += 1
            reward = 10
            self.place_food()
        else :
            self.snake.pop()
        
        self.update_ui()
        self.clock.tick(GAME_SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, point=None) :
        if point is None :
            point = self.head
        
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0 :
            return True
        
        if point in self.snake[1:] :
            return True
        
        return False
        
    def update_ui(self) :
        self.display.fill(BLACK)
        
        for point in self.snake :
            pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(point.x + 4, point.y + 4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()
        
    def move(self, action) :
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1,0,0]) :
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0,1,0]) :
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action, [0,0,1]) :
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT :
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT :
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN :
            y += BLOCK_SIZE
        elif self.direction == Direction.UP :
            y -= BLOCK_SIZE
            
        self.head = Point(x,y)