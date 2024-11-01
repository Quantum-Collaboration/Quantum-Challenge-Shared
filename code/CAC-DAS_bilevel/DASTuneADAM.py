#This is the main code for the tuner.
#It uses gradient descent to optimize window size, shape and position simultaneously.
#Class take in sample function and dimension D.
#no momentum or previous samples used
import numpy as np



class Sampler:

    eps = 1e-8
    rms_beta = 0.99
    mom_beta = 0.0
    v = 1.0

    prev_xw = []
    prev_L = []
    prev_xsamp = []
    prev_ysamp = []

    CRN = True

    def __init__(self, sample, D, B):
        self.D = D
        self.sample = lambda  x,prop_samp,seed : sample(x, prop_samp, seed)


        self.xw = np.zeros(D)
        self.L = np.diag(np.ones(D))

        self.tot_samp  = 0
        self.dt = 0.1
        self.g = 0.02
        self.g_exp = 0.0
        self.kappa = 1.0


        self.dt0 = 0.05

        self.fit_est = 0.00001
        self.fit_var = 0.0
        self.fit_est_beta = 0.05
        self.curv_est = 0.000
        self.curv_est_beta = 0.05
        self.grad_est = np.zeros(D)
        self.grad_est_beta = 0.05

        self.delta_fit = 0.0
        self.delta_fit2 = 0.0
        self.fit_prev = 0.0

        # self.B = 300#50
        # self.Bmax = 300#50
        self.B = B
        self.Bmin = B


        self.best_fit = 0.01**2
        self.Hbest = 0
        self.Hmin = 0
        self.Havg = 0
        


        self.growth = 0.05


    def init_window(self, xw, L):
        self.xw = xw
        self.L = L

    def step(self, dt, B):
        self.tot_samp += B
        D = self.D
        #random samples (z = normalized coord, x = real coord)
        z_samp = np.random.randn(D, B)

        seed = range(B)
        if(self.CRN):
            z_samp[:, :B//2] = -z_samp[:, B//2 :]
            seed = [2*s % B for s in seed]


        x_samp = self.xw.reshape(D, 1) + np.dot(self.L, z_samp)
        #sample fitness function
        #y_samp = self.sample(x_samp, seed)

        fitness_beta = 0
        # fitness_beta = 1-np.max([1.0-2*self.tot_samp/self.tot_samp_max,0])
        # fitness_beta = 0.8 + fitness_beta*0.2

        # if (self.tot_samp // self.B) % 10 == 0:
        #   print('steps',self.tot_samp/self.tot_samp_max, fitness_beta, self.B)
        #print('steps',self.tot_samp/self.tot_samp_max, fitness_beta, self.B, self.best_fit)

        # print('xw', self.xw)
        # print('L', self.L)
        # print('x', x_samp)
        # print('dt', dt, self.dt0, self.v, self.best_fit)

        y_samp,H_best,Hmin,Pbest,obj_best,objv_best,sol_best = self.sample(x_samp, self.tot_samp/self.tot_samp_max, seed)

        best_fit_ = self.best_fit + 0.0
        self.best_fit = np.maximum(np.average(y_samp**2), self.best_fit*self.rms_beta)
        #RMS normalization
        # self.v = self.v*( (self.best_fit + 0.001) /(best_fit_ + 0.001))**2
        # self.v = self.best_fit + self.eps
        # self.v = self.v + self.eps

        # self.dt = self.dt0/np.sqrt(self.v)
        self.Hbest = H_best
        self.Hmin = np.min(Hmin)
        self.Havg = np.mean(Hmin)
        self.obj_best = obj_best
        self.objv_best = objv_best
        
        self.dt = self.dt0 / np.sqrt(self.best_fit + self.eps)
        dt = self.dt

        #differentials in normalized coordinates
        dz = np.average(z_samp*y_samp.reshape(1,B), axis = 1)
        dA = -np.diag(np.ones(D))*np.average(y_samp) + np.average(z_samp.reshape(D, 1, B)*z_samp.reshape(1, D, B)*y_samp.reshape(1, 1,B), axis = 2)

        dz = dz*(1 - self.mom_beta)
        dA = dA*(1 - self.mom_beta)

        #save info for momentum
        self.prev_xw.append(self.xw)
        self.prev_L.append(self.L)
        self.prev_xsamp.append(x_samp)
        self.prev_ysamp.append(y_samp)


        #L_inv = np.linalg.inv(self.L)
        Lamb = np.dot(self.L.T, self.L)

        scale = np.trace(self.L)
        dxw =  np.dot(self.L, dz)*1

        dL =  np.dot(self.L, dA)*1/D

        #additional factor r is used to help with numerical stability when updating L.
        #This steps keeps the window from shrinking too quickly.
        L_ = self.L + dt*dL
        r = np.sum(L_**2)**0.5/np.sum(self.L**2)**0.5
        self.L = self.L + r*dt*dL

        #ensure step is not too big
        xw_step = r*dt*dxw
        zw_step = np.linalg.solve(self.L, xw_step)
        #expansion of sampling window (experimental)
        #self.L = self.L  + dt*self.g_current*np.diag(np.ones(D))
        self.L = self.L

        if(np.average(self.L**2)**0.5 < self.g_current):
            self.L =  self.L*self.g_current/np.average(self.L**2)**0.5


        #max size of window
        if(np.sqrt(np.average(self.L**2)) > 2):
            self.L =  self.L*2/np.sqrt(np.average(self.L**2 ))

        fest = np.average(y_samp)
        cest = np.average(y_samp*np.average(z_samp**2, axis = 0))
        gest = np.average(y_samp.reshape(1,B)*z_samp.reshape(D,B), axis = 1)

        self.fit_est = (1 - self.fit_est_beta)*self.fit_est + self.fit_est_beta*fest
        self.curv_est = (1 - self.curv_est_beta)*self.curv_est + self.curv_est_beta*cest
        self.grad_est = (1 - self.grad_est_beta)*self.grad_est + self.grad_est_beta*gest

        fvest = np.std(y_samp)**2/B
        #print(np.average(y_samp), np.std(y_samp), fvest**0.5)
        self.fit_var = (1 - self.fit_est_beta)*self.fit_var + self.fit_est_beta*np.sqrt(fvest)

        dfit = self.fit_est - self.fit_prev
        self.delta_fit = (1 - self.fit_est_beta)*self.delta_fit + self.fit_est_beta*dfit
        self.delta_fit2 = (1 - self.fit_est_beta)*self.delta_fit2 + self.fit_est_beta*dfit**2

        self.fit_prev = self.fit_est
        #ensure step is not too big
        #print("hello", self.grad_est)
        r2 = np.minimum(1.0, 1.0/np.linalg.norm(zw_step))
        #print("r2", r2)
        #update xw
        self.xw = self.xw + r2*xw_step



    def optimize(self, tot_samp_max = 50000, tr_min = 0, R_end = 10):
        self.tot_samp  =0
        self.tot_samp_max = tot_samp_max
        tot_samp_rec = []
        xw_rec = []
        L_rec = []
        fit_rec = []
        Hbest_rec = []
        obj_best_rec = []
        objv_best_rec = []
        Hmin_rec = []
        Havg_rec = []

        count = 0
        print(self.tot_samp , tot_samp_max, np.abs(np.sum(self.L**2)), (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est , R_end)
        while(self.tot_samp < tot_samp_max and np.abs(np.sum(self.L**2)) > tr_min and (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est < R_end):
            #batch size chosen as 1/trace(L) here so more accuracy when window shrinks

            R = (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est

            if(count > 20):
                if((self.fit_var)/self.fit_est > 4*(1 - R)):
                    self.B += 2
                    #self.fit_var = 0.
                elif((self.fit_var)/self.fit_est < 1*(1 - R)):
                    self.B -= 2
                if(self.B < self.Bmin):
                    self.B = self.Bmin



            self.g_current = self.g/(count + 1)**self.g_exp
            #B = 4

            #print("dt", np.exp(self.dt_log), np.log(1/self.fit_est), self.fit_est)
            # self.step(self.dt0/np.sqrt(self.v), self.B)
            self.step(self.dt, self.B)

            #print (for debug)
            # if(True and count % 20 == 0):
            #   print(self.B)
            #   print("x", self.xw)

            #   print("dt", self.dt0/np.sqrt(self.v), self.fit_est)
            #   print(np.diag(np.dot(self.L, self.L.T))**0.5)
            #   print("var ", self.fit_var, "var r", (self.fit_var)/self.fit_est)

            #save info
            tot_samp_rec.append(self.tot_samp)
            xw_rec.append(self.xw)
            L_rec.append(self.L)
            fit_rec.append(self.best_fit)
            Hbest_rec.append(self.Hbest)
            obj_best_rec.append(self.obj_best)
            objv_best_rec.append(self.objv_best)
            Hmin_rec.append(self.Hmin)
            Havg_rec.append(self.Havg)
            count += 1

        if(R > R_end):
            print("tuning terminated due to curvature condition R = " + str((self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est))


        return  tot_samp_rec, xw_rec, L_rec, fit_rec, Hbest_rec, obj_best_rec, objv_best_rec, Hmin_rec, Havg_rec



