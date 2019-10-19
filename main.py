# import matplotlib
# matplotlib.use("Agg")
import BatAlgorithm2 as ba
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# for i in range(10):
#     Algorithm = ba.BatAlgorithm(2, 200, 100, 0.5, 0.5, 0.0, 2.0, -5.0, 5.0, ba.ackley)
#     Algorithm.move_bat()

showanimation = True
### BatAlgorithm(dim, pop_size, max num of gen, loudness, pulse rate, freq_min, freq_max, l_bound, u_bound, functionname)

Algorithm = ba.BatAlgorithm(2, 200, 300, 0.5, 0.5, 0.0, 2.0, -5.0, 5.0, ba.rosenbrock)

if showanimation == False:
    Algorithm.move_bat()

else:
    result = Algorithm.move_bat()
    Writer = animation.writers['html']
    writer = Writer(fps=1, metadata=dict(artist='Me'))

    fig = plt.figure(figsize=(16, 12))
    ax = Axes3D(fig)
    scatter, = plt.plot([], [], [], 'bo', color='#03006b', zorder = 10)
    genn = plt.title('')
    text = ax.text(7.5, 2, 0, "f")
    minx1 = ax.text(7.5, 0, 0, "x1")
    minx2 = ax.text(7.5, -2, 0, "x2")

    def init():
        Algorithm.plot3d(fig,ax)
        return scatter,

    def update(i):
        pop, bestfitness, xbest = next(result)
        n1=[]
        n2=[]
        zz=[]
        for x in pop:
            n1.append(x[0])
            n2.append(x[1])
            zz.append(Algorithm.fitness(x))
        scatter.set_data(n1, n2)
        scatter.set_3d_properties(zz)
        genn.set_text('Iteration...%d' % i)
        text.set_text('f=%.10f' % bestfitness)
        minx1.set_text('x1=%.10f' % xbest[0])
        minx2.set_text('x2=%.10f' % xbest[1])
        return scatter, genn, text, minx1, minx2,
    ani = animation.FuncAnimation(fig, update, frames = 300, init_func=init, blit=True)
    plt.show()

    #ani.save('bat-rastrigin.html', writer=writer)