import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import time

class BatAlgorithm():
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function):
        self.D = D  # dimension
        self.NP = NP  # population size
        self.N_Gen = N_Gen  # generations
        self.A = A  # loudness (typical initial loudness [1,2]
        self.r = r  # pulse rate [0,1]
        self.Qmin = Qmin  # frequency min 0
        self.Qmax = Qmax  # frequency max 100 ### depends on the domain
        self.Lower = Lower  # lower bound
        self.Upper = Upper  # upper bound
        self.f_min = 0.0  # minimum fitness
        self.Lb = [0] * self.D  # lower bound
        self.Ub = [0] * self.D  # upper bound
        self.Q = [0] * self.NP  # frequency
        self.v = [[0 for i in range(self.D)] for j in range(self.NP)]  # velocity
        self.Sol = [[0 for i in range(self.D)] for j in range(self.NP)]  # population of solutions
        self.Fitness = [0] * self.NP  # fitness
        self.best = [0] * self.D  # best solution
        self.Fun = function
        self.ByFitAvg = True
        self.AverageF = []
        self.AvgFCoincide = 0
        self.MaxFitAvgC = 0.1 * self.N_Gen
        self.rpulse = [self.r] * self.NP
        self.Al = [self.A] * self.NP
    def best_bat(self):
        i = 0
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]
    def init_bat(self):
        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper
        for i in range(self.NP):
            self.Q[i] = 0
            for j in range(self.D):
                rnd = random.uniform(0, 1)
                self.v[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd
            self.Fitness[i] = self.Fun(self.Sol[i])
        self.best_bat()
    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val
    def move_bat(self):
        stime = time.time()
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]
        self.init_bat()
        for t in range(self.N_Gen):
            for i in range(self.NP):
                rnd = random.uniform(0, 1)
                self.Q[i] = self.Qmin + (self.Qmin - self.Qmax) * rnd
                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] -
                                                   self.best[j]) * self.Q[i]
                    S[i][j] = self.Sol[i][j] + self.v[i][j]
                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                self.Ub[j])
                rnd = random.random()
                if rnd > self.r:
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                    self.Ub[j])
                Fnew = self.Fun(S[i])
                rnd = random.random()
                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew
                    # if i>=1:
                    #     self.Al[i] = self.Al[i-1] * 0.9
                    # self.rpulse[i] = self.r * (1 - np.exp(-0.9*t))
                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew
            self.AverageF.append(sum(self.Fitness)/self.NP)
            if t >= 1 and self.AverageF[t] == self.AverageF[t-1]:
                self.AvgFCoincide = self.AvgFCoincide + 1
            if self.ByFitAvg == True:
                if self.AvgFCoincide == self.MaxFitAvgC:
                    print("Convergence at iteration...", t)
                    break
            yield self.Sol, self.f_min, self.best
        etime = time.time()
        print("f = {0:.20f}".format(self.f_min))
        for i in range(self.D):
            print("x{}".format(i), "=", "%.20f" % self.best[i])
        print("******************************")
        print("Time taken:", etime - stime)
    def plot3d(self, fig, ax, points=100):
        x = np.linspace(self.Lower, self.Upper, points)
        y = np.linspace(self.Lower, self.Upper, points)
        X, Y = np.meshgrid(x, y)
        solution = [X, Y]
        Z = self.Fun(solution)
        surf = ax.plot_surface(X, Y, Z, rstride=1,
                               cstride=1,
                               linewidth=0.1,
                               edgecolors='k',
                               alpha=0.5,
                               antialiased=True,
                               cmap=cm.RdPu_r, zorder=1)
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.01f'))

    def fitness(self, solution):
        return self.Fun(solution)

def rosenbrock(solution): ### -5; 5
    solution = np.asarray(solution)
    return sum(100.0*(solution[1:]-solution[:-1]**2.0)**2.0 + (1-solution[:-1])**2.0)

def ackley(solution): ### -5; 5
    solution = np.asarray(solution)
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(solution)
    s1 = sum(np.power(solution, 2))
    s2 = sum(np.cos(c * solution))
    return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)

def rastrigin(solution): ### -5; 5
    solution = np.asarray(solution)
    a = 10
    d = len(solution)
    s = np.power(solution, 2) - a * np.cos(2 * np.pi * solution)
    return a * d + sum(s)

def levy(solution): ### -5; 5
    solution = np.asarray(solution)
    w = 1 + (solution - 1) / 4
    wp = w[:-1]
    wd = w[-1]
    a = np.sin(np.pi * w[0]) ** 2
    b = sum((wp - 1) ** 2 * (1 + 10 * np.sin(np.pi * wp + 1) ** 2))
    c = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)
    return a + b + c

def eggholder(x): ### -512, 512 ### only 2d
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

def schwefel(x): ### -500, 500
    x = np.asarray(x)
    d = len(x)
    return 418.9829*d - sum(x*np.sin(np.sqrt(np.abs(x))))

def griewank(x): ### -600, 600
    x = np.asarray(x)
    a = sum(x ** 2 / 4000)
    b = 1
    for i in range(len(x)):
        b *= np.cos(x[i] / np.sqrt(i + 1))
    return a - b + 1

def dixonprice(x): ### -10, 10
    x = np.asarray(x)
    c = 0
    for i in range(1, len(x)):
        c += i * (2 * x[i] ** 2 - x[i - 1]) ** 2
    return (x[0] - 1) ** 2 + c

def michalewicz(x): ### 0, np.pi
    x = np.asarray(x)
    m = 10
    c = 0
    for i in range(0, len(x)):
        c += np.sin(x[i]) * np.sin(((i + 1) * x[i] ** 2) / np.pi) ** (2 * m)
    return -c

def easom(x): ### - 100, 100 ### only 2d
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

def styblinski_tang(x): ### -5, 5
    x = np.asarray(x)
    return sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2