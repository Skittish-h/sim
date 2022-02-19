LIN_IN = False
GRAPHIC = True
FRAMERATE = 30


import csv
import pygame
import math
import random
import numpy as np
from pathlib import Path

class vec2:
    x: float
    y: float
    def get_duplicate(self):
        return vec2(self.x, self.y)

    def __init__(self, x ,y, angle_f = False) -> None:
        if not angle_f:
            self.x = x
            self.y = y
        else:
            self.x = y * math.cos(x)
            self.y = y * math.sin(x)
    def get_angle(self):
        return math.atan2(self.y, self.x)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def multiply(self, factor):
        self.x = self.x*factor
        self.y = self.y*factor

    def scale_to_magnitude(self, magnitude):
        desired_factor = magnitude / self.magnitude()
        self.multiply(desired_factor)





class crd:
    x: float
    y: float
    def __init__(self) -> None:
        pass
    def __init__(self,x, y) -> None:
        self.x = x
        self.y = y
    
    def distance_to(self, point):
        return math.sqrt((point.x - self.x)**2 + (point.y - self.y)**2)
    
    def vec_to(self, point):
        return vec2((point.x - self.x), (point.y - self.y))
    
    def to_tuple(self, factor):
        transformed = self.scale(factor)
        return (transformed.x, transformed.y)

    def to_inv_tuple(self, factor):
        transformed = self.scale(factor)
        return (transformed.x, factor - transformed.y)

    def scale(self, factor):
        return crd(self.x*factor, self.y*factor)
    
    def apply_vec(self, vec: vec2):
        self.x = self.x + vec.x
        self.y = self.y + vec.y
    
    def add_vec(self, vec:vec2):
        cord = crd(0, 0)
        cord.x = self.x + vec.x
        cord.y = self.y + vec.y 

        return cord

class ball:
    coordinate: crd
    velocity: vec2

    def __init__(self, x, y, v_x, v_y) -> None:
        self.coordinate = crd(x, y)
        self.velocity = vec2(v_x, v_y)
    
    def apply_speed(self):
        self.coordinate.apply_vec(self.velocity)
    
    def ball_t(self, t):
        traveled_vec = self.velocity.get_duplicate()
        traveled_vec.multiply(t)

        
        return self.coordinate.add_vec(traveled_vec) 


class robot: 
    coordinate: crd
    velocity: vec2
    
    def __init__(self, x, y, max_speed) -> None:
        self.coordinate = crd(x, y)
        self.velocity = vec2(0, 0)
        self.max_speed = max_speed
    
    def apply_velocity(self: vec2):
        
        self.coordinate.apply_vec(self.velocity)
    
    def set_velocity(self, velocity):
        self.velocity = velocity
        if (self.velocity.magnitude() > self.max_speed):
            self.velocity.scale_to_magnitude(self.max_speed)



def proportional(x):
    return x

class simulation:
    def __init__(self) -> None:
        pygame.init()
        if LIN_IN:
            self.error_sum = 0
            self.last_i = False
        self.past_functions = []
        self.factor = 900
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.factor, self.factor))

        self.fle = Path('data.csv')
        self.fle.touch(exist_ok=True)
        

        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        self.font = pygame.font.SysFont('Arial', 20)

        rand_dir = random.random()*3.14*2

        random_x = math.cos(rand_dir)*0.002
        random_y = math.sin(rand_dir)*0.002


        self.ball = ball(0.5, 0.5, random_x, random_y)
        self.robot = robot(0.8, 0.5, 0.006)
        self.target = crd(0, 0.5)
    
    def reset(self):
        rand_dir = random.random()*3.14*2

        random_x = math.cos(rand_dir)*0.002
        random_y = math.sin(rand_dir)*0.002

        self.ball = ball(0.5, 0.5, random_x, random_y)
        self.robot = robot(0.8, 0.5, 0.006)
        self.target = crd(0, 0.5)
                
        if LIN_IN:
            self.error_sum = 0
            self.last_i = False

    def handle_game_over_quad(self):
        
        if((self.ball.coordinate.x < 0) or (self.ball.coordinate.x > 1) or (self.ball.coordinate.y < 0) or (self.ball.coordinate.y > 1)):
            file = open(self.fle, "a")
            writer = csv.writer(file)
            error = self.target.y - self.ball.coordinate.y
            writer.writerow([error])
            file.close()
            print("Done, error: ", error)
            self.reset()

    def handle_game_over_lin(self):
        # gets called if there was a collision so assumes game is over
        file = open(self.fle, "a")
        writer = csv.writer(file)
        writer.writerow([self.error_sum])
        file.close()
        print("Done, error: ", self.error_sum)
        self.reset()

    def main_loop(self):    
        active = True
        do_anything = True
        # run until window is closed
        while active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    active = False
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_SPACE:
                        do_anything = not do_anything
                    

            if do_anything:
                # clear display
                self.screen.fill((0,0,0))

                self.compute_collisions()

                target = 0
                if LIN_IN:
                    intercept = self.compute_linear_intercept()
                    if not self.last_i:
                        self.last_i = intercept

                    if(intercept):
                        target = self.robot.coordinate.vec_to(intercept)
                        # self.draw_intercept(intercept)
                        self.robot.set_velocity(target)
                        self.error_sum += self.last_i.distance_to(intercept)
                        
                        self.robot.apply_velocity()
                    else:
                        target = self.robot.coordinate.vec_to(self.ball.coordinate)
                    self.last_i = intercept

                else:
                    target, ball = self.compute_quadratic_target()
                    self.draw_target(self.target)
                    if(target and ball):
                        
                        
                        target.multiply(-1)
                        target.scale_to_magnitude(self.robot.max_speed)
                        self.draw_tangent(ball)
                        self.draw_intercept(ball)
                        self.robot.set_velocity(target)
                        
                        self.robot.apply_velocity()

                self.draw_robot()
                self.draw_ball()

                self.print_debugs()
                if LIN_IN:
                    pass
                else:
                    self.handle_game_over_quad()

                # update the display, and chill until its the time for the next frame
                pygame.display.update()
            self.clock.tick(FRAMERATE)


    def draw_intercept(self, intercept):
        pygame.draw.circle(self.screen, (255,50,50), self.to_pygame(intercept.to_tuple(self.factor)), 8, 8)

    def draw_target(self, intercept):
        pygame.draw.circle(self.screen, (255,0,255), self.to_pygame(intercept.to_tuple(self.factor)), 15, 15)
    
    def draw_tangent(self, ball_t: crd):
        pygame.draw.line(self.screen, (255,255,255), self.target.to_tuple(self.factor), ball_t.to_inv_tuple(self.factor))

    def draw_ball(self):
        # move ball
        self.ball.apply_speed()
        self.draw_vector(self.ball.coordinate, self.ball.velocity, (255,255,255))
        pygame.draw.circle(self.screen, (255,255,255), self.to_pygame(self.ball.coordinate.to_tuple(self.factor)), 8, 8)

    def draw_robot(self):
        
        pygame.draw.circle(self.screen, (255,255,0), self.to_pygame(self.robot.coordinate.to_tuple(self.factor)), 18, 3)
        self.draw_vector(self.robot.coordinate, self.robot.velocity, (255,255,0))
        textsurface = self.font.render("R", False, (255, 255, 0))
        text_crd = list(self.robot.coordinate.to_tuple(self.factor))

        text_crd[1] = self.factor - text_crd[1] - 8
        text_crd[0] = text_crd[0]- 6
        self.screen.blit(textsurface, tuple(text_crd))
    def draw_vector(self, point: crd, vector: vec2, color):
        vector = vector.get_duplicate()
        vector.multiply(12)
        pygame.draw.line(self.screen, color, point.to_inv_tuple(self.factor), point.add_vec(vector).to_inv_tuple(self.factor), 1)

        angle = vector.get_angle()
        angle1 = angle + (3*math.pi)/4
        angle2 = angle - (3*math.pi)/4

        arrow_vector1 = vec2(angle1, 0.01, angle_f=True)
        arrow_vector2 = vec2(angle2, 0.01, angle_f=True)
        pygame.draw.line(self.screen, color, point.add_vec(vector).to_inv_tuple(self.factor), point.add_vec(vector).add_vec(arrow_vector1).to_inv_tuple(self.factor))
        pygame.draw.line(self.screen, color, point.add_vec(vector).to_inv_tuple(self.factor), point.add_vec(vector).add_vec(arrow_vector2).to_inv_tuple(self.factor))


    def print_debugs(self):
        # print debug messages
        displaystring= ""
        displaystring += f"ball X: {self.ball.coordinate.x:.2f}u  "
        displaystring += f"ball Y: {self.ball.coordinate.y:.2f}u  "
        displaystring += f"ball V_x: {self.ball.velocity.x:.4f}u/s  "
        displaystring += f"ball V_x: {self.ball.velocity.y:.4f}u/s  "
        displaystring2 = f"robot X: {self.robot.coordinate.x:.2f}u  "
        displaystring2 += f"robot Y: {self.robot.coordinate.y:.2f}u  "
        displaystring2 += f"robot V_x: {self.robot.velocity.x:.4f}u/s  "
        displaystring2 += f"robot V_x: {self.robot.velocity.y:.4f}u/s  "
        textsurface = self.font.render(displaystring, False, (255, 255, 255))
        textsurface2 = self.font.render(displaystring2, False, (255, 255, 255))

        self.screen.blit(textsurface, (10, 10))
        self.screen.blit(textsurface2, (10, 40))

    def compute_linear_intercept(self):
        # using things computed in IA
        a = (self.ball.velocity.x**2) + (self.ball.velocity.y**2) - (self.robot.max_speed**2)
        b = 2*((self.ball.velocity.x*self.ball.coordinate.x) - (self.robot.coordinate.x*self.ball.velocity.x) + (self.ball.velocity.y*self.ball.coordinate.y) - (self.robot.coordinate.y*self.ball.velocity.y))
        c = ((self.ball.coordinate.x**2) - (2*self.ball.coordinate.x*self.robot.coordinate.x) + (self.robot.coordinate.x**2) + (self.ball.coordinate.y**2) - (2*self.ball.coordinate.y*self.robot.coordinate.y) + (self.robot.coordinate.y**2))

        # construct polynomial in numpy
        poly = np.poly1d([a, b, c])
        if poly.r.any():
            min_sol = 10e7
            for solution in poly.r:
                if solution > 0 and solution < min_sol and np.isreal(solution):
                    min_sol = solution
            if min_sol != 10e7:
                target = self.ball.ball_t(min_sol)
                return target
                    
        # get to roots of polynomial

        return 0
    def compute_quadratic_target(self):
        ITERATONS = 1000
        STEP = 1
        MULTIPLE = 0.1
        MAX_NUM_PARAS = 20
        
        parabola_selector = 4

        # for every timestep
        for time in range(0, ITERATONS, STEP):
            time *= MULTIPLE
            coefs, ball_t = self.get_quadratic_coefficients(time)
            
            arc_length = self.get_parabolic_arc_length(coefs, self.robot.coordinate.x) - self.get_parabolic_arc_length(coefs, ball_t.x)


            display = True

            if GRAPHIC and not (time % parabola_selector):
                quad = lambda x: coefs[0]*(x**2) + coefs[1]*x + coefs[2]
                self.draw_function(quad, int(ball_t.scale(self.factor).x), int(self.robot.coordinate.scale(self.factor).x), 1, (0,255,0))


            # print(arc_length.real)
            
            if arc_length.real < self.robot.max_speed * time:
                derivative = 2*coefs[0]*self.robot.coordinate.x + coefs[1]*self.robot.coordinate.x
                dir_vector = vec2(1, derivative)
                if display:
                    quad = lambda x: coefs[0]*(x**2) + coefs[1]*x + coefs[2]
                    self.draw_function(quad, int(ball_t.scale(self.factor).x), int(self.robot.coordinate.scale(self.factor).x), 1, (255,0,0))

                return dir_vector, ball_t
        return None, None

    #parabolic arc length from 0 to "x" of a parabola with coeffcients a, b, c
    def get_parabolic_arc_length(self, coefficients, x):
        a,b,c = tuple(coefficients)

        # ugly formula computed in IA

        # ((b + 2 a x) Sqrt[1 + (b + 2 a x)^2] + ArcSinh[b + 2 a x])/(4 a)
        return (math.sqrt((2*a*x + b)**2 + 1) * (2*a*x + b) + math.asinh(2*a*x + b)) / (4*a)



    def get_quadratic_coefficients(self, time):
        # need to build the system of equations for quadratic coefficiencts
        ball_t = self.ball.ball_t(time)
        # print(ball_t.x, ball_t.y, time)
        # eq 1: ball passes through robot position
        eq1 = [self.robot.coordinate.x**2, self.robot.coordinate.x, 1, self.robot.coordinate.y]

        # eq2: ball passes through future ball position
        eq2 = [ball_t.x**2, ball_t.x, 1, ball_t.y]

        desired_vector = ball_t.vec_to(self.target)
        
        # ensure no null division occurs
        if desired_vector.x ==0:
            desired_vector.x = 1e-5

        dydx = desired_vector.y/desired_vector.x

        # derovative at ball_x goes to goal
        eq3 = [2*ball_t.x, 1, 0, dydx]

        # formulate matrixes
        A = np.array([eq1[0:3], eq2[0:3], eq3[0:3]])
        B = np.array([eq1[3], eq2[3], eq3[3]])

        return (np.linalg.solve(A,B), ball_t)


    def compute_collisions(self):
        COLLISION_THRESHOLD = 0.01
        if(self.ball.coordinate.distance_to(self.robot.coordinate) < COLLISION_THRESHOLD):
            # collision has occured. Simplyfied collision model ensues:
            # just double whatever the velocity of the robot is before impact
            target = self.robot.velocity.get_duplicate()
            target.multiply(2)
            self.ball.velocity = target
            if LIN_IN:
                self.handle_game_over_lin()

    def to_pygame(self, coords):
        """Convert coordinates into pygame coordinates (lower-left => top left)."""
        return (coords[0], self.factor - coords[1])

    def draw_function(self, func, x_min, x_max, thickness, color):
        if not( x_max > x_min):
            xm_tmp = x_max
            x_max = x_min
            x_min = xm_tmp
        for x in range(x_min, x_max):
            normal_x = x/self.factor
            y = func(normal_x)
            target = crd(normal_x, 1-y)
            p_x, p_y = target.to_tuple(self.factor)
            for i in range(thickness):
                

                
                self.screen.set_at((int(p_x), int(p_y)+i), color)


sim = simulation()
sim.main_loop()