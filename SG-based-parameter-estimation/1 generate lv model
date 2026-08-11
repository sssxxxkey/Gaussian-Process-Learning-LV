import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

np.random.seed(42)

def LV_model(x0,y0,total_time,dt,params):

    alpha,beta,delta,gamma=params

    def lv(t,state):
        x,y=state

        dx=alpha*x-beta*x*y
        dy=delta*x*y-gamma*y

        return [dx,dy]


    t=np.linspace(
        0,
        total_time,
        int(total_time/dt)+1
    )


    sol=solve_ivp(
        lv,
        (0,total_time),
        [x0,y0],
        t_eval=t
    )

    return sol.y[0],sol.y[1],sol.t
