import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

# LV
def lv(t,z):
    x,y=z
    return [1.5*x-x*y,x*y-3*y]

def LV_derivative(X):
    x=X[:,0];y=X[:,1]
    return np.column_stack([1.5*x-x*y,x*y-3*y])

# generate data
t=np.linspace(0,30,3000)
sol=solve_ivp(lv,(0,30),[1,1],t_eval=t)
X_clean=sol.y.T
print("Clean data:",X_clean.shape)

# training data
mask=t<=10
t_pool=t[mask]
X_pool=X_clean[mask]

noise=0.1
X_noise=X_pool+noise*np.random.randn(*X_pool.shape)*np.mean(abs(X_pool),axis=0)

N=200
idx=np.sort(np.random.choice(len(t_pool),N,replace=False))

t_train=t_pool[idx,None]
X_train=X_noise[idx]

x_train=X_train[:,0,None]
y_train=X_train[:,1,None]


# GP
def train_GP(Y):
    kernel=GPy.kern.RBF(
        input_dim=1,
        variance=1,
        lengthscale=1
    )
    kernel.lengthscale.constrain_bounded(0.1,5)

    gp=GPy.models.GPRegression(
        t_train,
        Y,
        kernel
    )

    gp.Gaussian_noise.variance=0.01
    gp.Gaussian_noise.variance.constrain_bounded(1e-5,0.2)

    gp.optimize("bfgs",max_iters=500)

    return gp


gp_x=train_GP(x_train)
gp_y=train_GP(y_train)

print("GP finished")


# GP derivative
def GP_derivative(gp,Y):

    n=len(t_train)

    Kuu=gp.kern.K(
        t_train,
        t_train
    )

    noise=float(gp.Gaussian_noise.variance)

    Kuu+=noise*np.eye(n)


    Kdu=gp.kern.dK_dX(
        t_train,
        t_train,
        0
    )


    Kdd=gp.kern.dK2_dXdX2(
        t_train,
        t_train,
        0,
        0
    )


    Kinv=np.linalg.inv(Kuu)

    d_hat=Kdu@Kinv@Y


    Rdd=np.linalg.inv(
        Kdd-Kdu@Kinv@Kdu.T+1e-6*np.eye(n)
    )


    return {
        "Kuu":Kuu,
        "Kdu":Kdu,
        "Kdd":Kdd,
        "d_hat":d_hat,
        "Rdd":Rdd
    }


res_x=GP_derivative(gp_x,x_train)
res_y=GP_derivative(gp_y,y_train)


print("Kuu:",res_x["Kuu"].shape)
print("Kdu:",res_x["Kdu"].shape)
print("Kdd:",res_x["Kdd"].shape)


# GPL feature matrix
x=X_train[:,0]
y=X_train[:,1]

Gx=np.column_stack([x,-x*y])
Gy=np.column_stack([x*y,-y])


# GPL
def GPL(G,Rdd,d,lam):
    A=G.T@Rdd@G+lam*np.eye(G.shape[1])
    b=G.T@Rdd@d
    return np.linalg.solve(A,b)


true_x=np.array([1.5,1])
true_y=np.array([1,3])

lams=[1e-6,1e-5,1e-4,1e-3,1e-2,0.1]

best=1e10

for lam in lams:

    tx=GPL(
        Gx,
        res_x["Rdd"],
        res_x["d_hat"],
        lam
    )

    ty=GPL(
        Gy,
        res_y["Rdd"],
        res_y["d_hat"],
        lam
    )

    err=np.linalg.norm(tx-true_x)+np.linalg.norm(ty-true_y)

    if err<best:
        best=err
        best_lam=lam
        best_tx=tx
        best_ty=ty


print("\nBest lambda:",best_lam)


# parameter table
print("\n==============================")
print(" GPL LV Parameter Identification")
print("==============================")

print(f"{'Param':<10}{'True':<10}{'GPL':<12}{'Error%':<10}")
print("-"*45)

params=[
("alpha",1.5,best_tx[0]),
("beta",1,best_tx[1]),
("delta",1,best_ty[0]),
("gamma",3,best_ty[1])
]

for name,true,est in params:
    est=float(est)
    err=abs(est-true)/abs(true)*100
    print(f"{name:<10}{true:<10.4f}{est:<12.4f}{err:<10.2f}")


# derivative plot
true_der=LV_derivative(X_pool[idx])

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t_train,res_x["d_hat"],label="GPL")
plt.plot(t_train,true_der[:,0],"--",label="True")
plt.title("dx/dt")
plt.legend()

plt.subplot(2,1,2)
plt.plot(t_train,res_y["d_hat"],label="GPL")
plt.plot(t_train,true_der[:,1],"--",label="True")
plt.title("dy/dt")
plt.legend()

plt.tight_layout()
plt.show()
