# bat-algorithm
Evolutionary computation algorithm called Bat algorithm + test functions animation of the algorithm progress by iteration

# Short decription
Updated code by https://github.com/buma/BatAlgorithm, added test functions animations as plot3d function in the main class, added main.py for running tests. In order to view animations, set variable `showanimation = True` and in the init file uncomment `yield`. For proper animation number of frames should correspond to the number of algorithm iterations.
Additionally animations can be saved as files in either html format (each frame will be saved as an image in the set folder for further html animation generation) or mp4. To save an animation comment `plt.show()` and uncomment `ani.save`. The animation is demonstated below (Ackley function):

![ackleygif](https://github.com/natuthepatatu/bat-algorithm/blob/master/ackley.gif)

Rosenbrock function:
![rosenbrock](https://github.com/natuthepatatu/bat-algorithm/blob/master/rosenbrock.jpg)

Rastrigin function:
![rastrigin](https://github.com/natuthepatatu/bat-algorithm/blob/master/rastrigin.jpg)

Overall there are 10 test functions:
* rosenbrock
* ackley
* rastrigin
* levy
* eggholder
* schwefel
* griewank
* dixonprice
* michalewicz
* easom
* styblinski_tang
