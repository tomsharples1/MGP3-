import numpy as np 
import matplotlib.pyplot as plt 

class Animation_1D:

    def __init__(self, length=50, time=5, nodes=30, D_T=0.1068, beta=0.0411, 
                 c=0.001, a=100, initial_temp=27, boundary_temp=200):
        
        # varibales
        self.length = length
        self.time = time 
        self.nodes = nodes 
        self.D_T = D_T
        self.beta = beta 
        self.c = c
        self.a = a
        self.initial_temp = initial_temp 
        self.boundary_temp = boundary_temp

        # Deltas
        self.dx = self.length / self.nodes
        self.dt = (self.dx**2)/ 200

        # Initial conditions
        self.u = np.zeros(self.nodes) + self.initial_temp
    
        # Boundary Conditions 
        self.u[0] = self.boundary_temp 
        self.u[-1] = self.boundary_temp 

        # Counters, flags and empty arrays 
        self.counter = 0
        self.blow_up_flag = False
        self.break_counter = 0
        self.avg_temp = []
        self.time_list = []
        self.frame_number = 0 
        
    def simulate(self):

        self.fig, self.axis = plt.subplots()
        self.pcm = self.axis.pcolormesh([self.u], cmap=plt.cm.jet, vmin=self.initial_temp, vmax=300) # plt.cm.jet is blue to red
        self.colorbar = plt.colorbar(self.pcm, ax=self.axis)
        self.axis.set_ylim([-1, 2])

        self.avg_temp_marker = self.colorbar.ax.plot([0.5], [0], marker='_', color='white', markersize=25)
        
        while self.counter < self.time:
            self.w = self.u.copy()

            for i in range(1, self.nodes - 1):
                # Fully-dimensionalised
                self.u[i] = self.w[i] + self.dt * self.D_T * (self.w[i - 1] - 2*self.w[i] + self.w[i+1]) / self.dx**2 \
                    + self.dt * (self.c * self.a * np.exp(self.beta * self.w[i]))
                # u[i+1] - u[i] /dt = D_T * u[i-1] - 2*u[i] + u[i+1] / dx**2 + c*a*exp(beta * u[i])
                
                if self.u[i] > 1e6 and not self.blow_up_flag:
                    self.blow_up_flag = True
                    self.pcm.set_cmap('hot')
                    self.pcm.set_array([self.u])
                    self.axis.text(5, 1.5, f"Blow up at t={self.counter:.3f}", color="red", fontsize=20)
                    self.break_counter = self.counter + 20*self.dt
                    break

            self.counter += self.dt

            avg_temp = np.mean(self.u[1:self.nodes-1])
            self.avg_temp_marker[0].set_ydata(avg_temp)

            if self.counter > self.break_counter and self.blow_up_flag is True:
                break

            self.pcm.set_array([self.u])
            self.axis.set_title("Time t: {:.3f} [s]".format(self.counter), fontsize=15)
            plt.pause(0.1)

        plt.show()

    def graph(self):

        while self.counter < self.time:
            self.w = self.u.copy()

            for i in range(1, self.nodes - 1):
                # Fully-dimensionalised 
                self.u[i] = self.w[i] + self.dt * self.D_T * (self.w[i - 1] - 2*self.w[i] + self.w[i+1]) / self.dx**2  \
                    + self.dt * (self.c * self.a * np.exp(self.beta * self.w[i]))

            avg_temp = np.average(self.u[1:self.nodes-1])
            self.avg_temp.append(avg_temp)
            self.time_list.append(self.counter)

            self.counter += self.dt

        plt.plot(self.time_list, self.avg_temp, color="red")
        plt.ylabel("Average Temperature [C]", fontsize=15)
        plt.xlabel("Time [s]", fontsize=15)
        plt.title("1D Average Temperature against Time \nInitial Temp {:.1f} C\nBoundary Temp {:.1f} C \nInitial Conc {:.1f} mol/m^3".format(self.initial_temp, self.boundary_temp, self.a), fontsize=15)
        plt.ylim(self.initial_temp, self.boundary_temp+100)
        plt.xlim(0)
        plt.show()

    def run_graph(self, boundary_temps):

        self.boundary_temps = boundary_temps
        avg_temp_all = []
        time_list_all = []
        
        for temp in boundary_temps:
            animation_1D = Animation_1D(time=self.time, boundary_temp=temp) 

            animation_1D.graph()
            avg_temp_all.append(animation_1D.avg_temp)
            time_list_all.append(animation_1D.time_list)
        
        plt.figure(figsize=(10, 6))

        for i, (avg_temp, time_list) in enumerate(zip(avg_temp_all, time_list_all)):
            plt.plot(time_list, avg_temp, label=f"Boundary Temp: {boundary_temps[i]} C")

        plt.ylabel("Average Temperature [C]", fontsize=15)
        plt.xlabel("Time [s]", fontsize=15)
        plt.title("1D Average Temperature vs. Time with Different Boundary Temperatures \nInitial Temperature {:.1f} C \nInitial Concentration {:.1f} mol/m^3".format(self.initial_temp, self.a), fontsize=15)
        plt.legend()
        plt.ylim(self.initial_temp, self.boundary_temp +100)
        plt.xlim(0, self.time)
        plt.show()

# length=50, time=5, nodes=30, D_T=1.068, beta=0.0211, c=0.001, a=100, initial_temp=27, boundary_temp=200
animation= Animation_1D(time=30, boundary_temp=400, D_T=0.1068, beta=0.0411, c=0.001, a=100) 
#animation.simulate()
#animation_1D.graph()
#animation_1D.run_graph(boundary_temps = [200, 300, 400, 500])

class Animation_2D:

    def __init__(self, length=50, time=5, nodes=30, D_T=1.068, beta=0.0411, 
                 c=0.001, a=100, initial_temp=27, boundary_temp=200):
        
        # Variables
        self.length = length
        self.time = time
        self.nodes = nodes
        self.D_T = D_T
        self.beta = beta
        self.c = c
        self.a = a 
        self.initial_temp = initial_temp
        self.boundary_temp = boundary_temp
        
        # Deltas
        self.dx = self.length / self.nodes 
        self.dy = self.length / self.nodes
        self.dt = self.dx**2 / 200 # should be very small 
        
        # Initilize Temp
        self.u = np.zeros((self.nodes, self.nodes)) + self.initial_temp
        
        # Boundary conditions 
        self.u[0, :] = self.boundary_temp
        self.u[-1, :] = self.boundary_temp 
        self.u[:, 0] = self.boundary_temp
        self.u[:, -1] = self.boundary_temp
        
        # Counters, flags and empty arrays 
        self.counter = 0
        self.blow_up_flag = False
        self.break_counter = 0 
        self.avg_temp = [] # for the .graph() function
        self.time_list = []

    def simulate(self):
        # Initialize figure 
        self.fig, self.axis = plt.subplots()
        self.pcm = self.axis.pcolormesh(self.u, cmap=plt.cm.jet, vmin=self.initial_temp, vmax=260) #plt.cm.jet
        self.colorbar = plt.colorbar(self.pcm, ax=self.axis)
        
        self.avg_temp_marker = self.colorbar.ax.plot([0.5], [0], marker='_', color='white', markersize=25)

        while self.counter < self.time:
            w = self.u.copy()
            for i in range(1, self.nodes-1):
                for j in range(1, self.nodes - 1):
                    # Fully-dimensionalised
                    dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j]) / self.dx**2
                    dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1]) / self.dy**2
                    self.u[i, j] = self.dt * self.D_T * (dd_ux + dd_uy) + w[i, j] + self.dt * (self.c * self.a * np.exp(self.beta * w[i, j]))
                    
                    if self.u[i, j] > 1e6 and not self.blow_up_flag:
                        self.blow_up_flag = True
                        self.pcm.set_cmap('hot')
                        self.pcm.set_array(self.u)
                        self.axis.text(10, 15, f"Blow up at t={self.counter:.3f}", color="red", fontsize=20)
                        self.u[i,j] = np.nan
                        #self.u[:, 0] = 1e6
                        #self.u[:, -1] = 1e6
                        self.break_counter = self.counter + 28*self.dt
                        break

            self.counter += self.dt

            avg_temp = np.mean(self.u[1:self.nodes-1, 1:self.nodes-1])
            self.avg_temp_marker[0].set_ydata(avg_temp)

            if self.counter > self.break_counter and self.blow_up_flag is True:
                break

            self.pcm.set_array(self.u)
            self.axis.set_title("2D Distribution at t: {:.3f} [s]".format(self.counter), fontsize=15)
            plt.pause(0.1)

        plt.show()

    def graph(self):

        while self.counter < self.time:
            w = self.u.copy()

            for i in range(1, self.nodes -1):
                for j in range(1, self.nodes -1 ):
                    # Fully-dimensionalised
                    dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j]) / self.dx**2
                    dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1]) / self.dy**2
                    self.u[i, j] = self.dt * self.D_T * (dd_ux + dd_uy) + w[i, j] + self.dt * (self.c * self.a * np.exp(self.beta * w[i, j]))

            avg_temp = np.average(self.u[1:self.nodes-1, 1:self.nodes-1])
            self.avg_temp.append(avg_temp)
            self.time_list.append(self.counter)

            self.counter += self.dt

        plt.plot(self.time_list, self.avg_temp, color="red")
        plt.ylabel("Average Temperature [C]", fontsize=15)
        plt.xlabel("Time [s]", fontsize=15)
        plt.title("2D Average Temperature against Time \nInitial Temp {:.1f} C\nBoundary Temp {:.1f} C \nInitial Conc {:.1f} mol/m^3".format(self.initial_temp, self.boundary_temp, self.a), fontsize=15)
        plt.ylim(0, self.boundary_temp + 100)
        plt.xlim(0)
        plt.show()

    def run_graph(self, boundary_temps):

        self.boundary_temps = boundary_temps
        avg_temp_all = []
        time_list_all = []

        timer = 40
        for temp in boundary_temps:
            animation_2D = Animation_2D(time=timer, boundary_temp=temp) 

            animation_2D.graph()
            avg_temp_all.append(animation_2D.avg_temp)
            time_list_all.append(animation_2D.time_list)

        plt.figure(figsize=(10,6))

        for i, (avg_temp, time_list) in enumerate(zip(avg_temp_all, time_list_all)):
            plt.plot(time_list, avg_temp, label=f"Boundary Temp: {boundary_temps[i]} C")

        plt.ylabel("Average Temperature [C]")
        plt.xlabel("Time [s]")
        plt.title("2D Average Temperature vs. Time with Different Boundary Temperatures")
        plt.legend()
        plt.ylim(20, 150)
        plt.xlim(0, timer)
        plt.show()

animation_2D = Animation_2D(time=50, length=50, nodes=30, boundary_temp=700, D_T=0.1068, beta=0.0411, c=0.001, a=100, initial_temp=27) 
#animation_2D.simulate()
#animation_2D.graph()
#animation_2D.run_graph([100, 200, 300, 400])

class Combined:

    def __init__(self, length=50, time=5, nodes=30, D_T=1.068, D_a = 0.1068, beta=0.0411, c=0.001,k=0.001,
                 initial_temp=27, initial_conc=100, boundary_temp=200):

        # Variables
        self.length = length
        self.time = time
        self.nodes = nodes
        self.D_T = D_T
        self.D_a = D_a
        self.c = c
        self.k = k
        self.beta = beta
        self.initial_temp = initial_temp
        self.initial_conc = initial_conc
        self.boundary_temp = boundary_temp
        
        # Initialise step sizes 
        self.dx = self.length / self.nodes
        self.dy = self.length / self.nodes
        self.dt = self.dx**2 / 200  # smaller the better 

        # Initialise data matricies
        self.u = np.zeros((self.nodes, self.nodes)) + self.initial_temp
        self.a_values = np.zeros((self.nodes, self.nodes)) + self.initial_conc

        # Boundary Conditions on temp
        self.u[0, :] = self.boundary_temp 
        self.u[-1, :] = self.boundary_temp
        self.u[:, 0] = self.boundary_temp
        self.u[:, -1] = self.boundary_temp
        
        #Counters, flags and empty arrays 
        self.counter = 0
        self.blow_up_flag = False
        self.zero_conc = False
        self.break_counter = 0
        self.avg_temp = []
        self.avg_conc = []
        self.time_list = []

    def simulate(self):
        
        # Create subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

        # temp pcolormesh
        self.pcm_u = self.ax1.pcolormesh(self.u, cmap=plt.cm.jet, vmin=self.initial_temp, vmax=260) # max(self.boundary_temp_1, self.boundary_temp_2)
        self.colorbar_u = plt.colorbar(self.pcm_u, ax=self.ax1) ##### change 
        self.ax1.set_title("Temperature at t: {:.3f} [s]".format(0))

        # conc pcolormesh 
        self.pcm_a = self.ax2.pcolormesh(self.a_values, cmap=plt.cm.plasma, vmin=0, vmax=self.initial_conc)
        self.colorbar_a = plt.colorbar(self.pcm_a, ax=self.ax2)
        self.ax2.set_title("Concentration of A at t: {:.3f} [s]".format(0))

        self.avg_temp_marker = self.colorbar_u.ax.plot([0.5], [0], marker='_', color='white', markersize=25)
        self.avg_conc_marker = self.colorbar_a.ax.plot([0.5], [0], marker='_', color='black', markersize=25)

        while self.counter < self.time:
            
            w = self.u.copy()
            a_w = self.a_values.copy()

            for i in range(1, self.nodes-1):
                for j in range(1, self.nodes-1):
                    
                    # Fully-dimensionalised
                    dd_ax = (a_w[i-1, j] - 2*a_w[i, j] + a_w[i+1, j]) / self.dx**2
                    dd_ay = (a_w[i, j-1] - 2*a_w[i, j] + a_w[i, j+1]) / self.dy**2
                    self.a_values[i, j] = a_w[i, j] + self.dt * self.D_a *  (dd_ax + dd_ay) \
                        - self.dt * (self.k * a_w[i, j] * np.exp(self.beta* w[i, j]))

                    # Fully-dimensionalised
                    dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j]) / self.dx**2
                    dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1]) / self.dy**2
                    self.u[i, j] = w[i, j] + self.dt * self.D_T * (dd_ux + dd_uy) \
                        + self.dt * (self.c * a_w[i,j] * np.exp(self.beta * w[i, j]))

                    if self.u[i, j] > 3000 and not self.blow_up_flag:
                        self.pcm_u.set_cmap('hot')
                        self.pcm_u.set_array(self.u)
                        self.ax1.text(5, 15, f"Blow up at t={self.counter:.3f}", color="red", fontsize=20)
                        self.ax2.text(3, 15, f"Concentration zero at t={self.counter:.3f}", color="blue", fontsize=15)
                        self.blow_up_flag = True
                        self.u[:, 0] = 1e6
                        self.u[:, -1] = 1e6
                        self.break_counter = self.counter + 30*self.dt
                        break

                    if np.average(self.a_values[1:self.nodes-1, 1:self.nodes-1]) <= 0.1 and not self.zero_conc:
                        self.ax2.text(3, 15, f"Concentration zero at t={self.counter:.3f}", color="blue", fontsize=15)
                        self.zero_conc = True
                        self.break_counter = self.counter + 30*self.dt
                        break

            self.counter += self.dt
            
            # avg temp for colorbar marker 
            avg_temp = np.mean(self.u[1:self.nodes-1, 1:self.nodes-1])
            self.avg_temp_marker[0].set_ydata(avg_temp) 

            # avg conc for colorbar marker 
            avg_conc = np.mean(self.a_values[1:self.nodes-1, 1:self.nodes-1])
            self.avg_conc_marker[0].set_ydata(avg_conc)

            if self.counter > self.break_counter and self.blow_up_flag is True:
                break

            if self.counter > self.break_counter and self.zero_conc is True:
                break
            
            self.pcm_u.set_array(self.u)
            self.pcm_a.set_array(self.a_values)
            self.ax2.set_title("Concerntration at t: {:.3f} [s] \nInitial Concentration of A: {:.1f}".format(self.counter, self.initial_conc))
            self.ax1.set_title("Temperature at t: {:.3f} [s] \nInitial Temperature {:.1f} C \nBoundary Temperatures {:.1f} C".format(self.counter, self.initial_temp, self.boundary_temp))
            plt.pause(0.1)

        plt.show()

    def graph(self, CorT=str):
        
        while self.counter < self.time:
            a_w = self.a_values.copy()
            w = self.u.copy()
            
            for i in range(1, self.nodes-1):
                for j in range(1, self.nodes-1):

                    # a_values following PDE
                    dd_ax = (a_w[i-1, j] - 2*a_w[i, j] + a_w[i+1, j]) / self.dx**2
                    dd_ay = (a_w[i, j-1] - 2*a_w[i, j] + a_w[i, j+1]) / self.dy**2
                    self.a_values[i, j] = a_w[i, j] + self.dt * self.D_a *  (dd_ax + dd_ay) \
                        - self.dt * (self.k * a_w[i, j] * np.exp(self.beta* w[i, j])) 

                    # u values following PDE 
                    dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j]) / self.dx**2
                    dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1]) / self.dy**2
                    self.u[i, j] = w[i, j] + self.dt * self.D_T * (dd_ux + dd_uy) \
                        + self.dt * (self.c * a_w[i,j] * np.exp(self.beta * w[i, j]))
                    
                    if self.u[i,j] > 1e6 and not self.blow_up_flag:
                        self.blow_up_flag = True
                        break

                    if np.mean(self.a_values[1:self.nodes-1, 1:self.nodes-1]) <= 0.01 and not self.zero_conc:
                        self.zero_conc = True
                        break


            # Creating a list for plotting temp vs time 
            avg_temp = np.average(self.u[1:self.nodes-1, 1:self.nodes-1])
            avg_conc = np.average(self.a_values[1:self.nodes-1, 1:self.nodes-1])
            self.avg_temp.append(avg_temp)
            self.avg_conc.append(avg_conc)
            self.time_list.append(self.counter)

            self.counter += self.dt

            if self.blow_up_flag is True or self.zero_conc is True:
                break 

        # Temp vs time
        if CorT == "temp":
            plt.plot(self.time_list, self.avg_temp, color='red')
            plt.xlabel('Time [s]', fontsize=15)
            plt.xlim(0, 5)
            plt.ylim(0, 4000)
            plt.ylabel('Average Temperature [C]', fontsize=15)
            plt.title('Average Temperature vs. Time \nInitial Temp {:.1f} ºC \nBoundary Temp {:.1f} ºC'.format(self.initial_temp, self.boundary_temp), fontsize=15)
            plt.show()

        # Conc vs time
        if CorT == "conc":
            plt.plot(self.time_list, self.avg_conc)
            plt.xlabel('Time [s]', fontsize=15)
            plt.xlim(0, self.time)
            plt.ylim(0, self.initial_conc) 
            plt.ylabel('Average Concerntration', fontsize=15)
            plt.title('Average Conc vs. Time \nInitial Conc {:.1f} mol/m^3'.format(self.initial_conc, self.boundary_temp), fontsize=15)
            plt.show()
        
        # Temp and Conc vs time
        if CorT == "both":
            plt.plot(self.time_list, self.avg_temp, color='red', label="Temperature")
            plt.plot(self.time_list, self.avg_conc, color='blue', label="Concentration")
            plt.legend(fontsize=15)
            plt.xlim(0, self.time)
            plt.ylim(-1, 200)
            plt.xlabel("Time [s]", fontsize=15)
            plt.ylabel("Avg temperature and Conc", fontsize=15)
            plt.title("Temperature and Conc on same graph against time \nInitial Temperature {:.1f} C\nBoundary Temperature {:.1f} C \nInitial Concentration {:.1f} mol/m^3".format(self.initial_temp, self.boundary_temp, self.initial_conc), fontsize=15)
            plt.show()

# length=50, time=5, nodes=30, D_T=1.068, D_a = 0.1068, beta=0.0411, c=0.001,k=0.001, initial_temp=27, initial_conc=100, boundary_temp=200
animation = Combined(time=200, D_T= 0.1068, D_a = 0.025, beta=0.0411, boundary_temp=200, initial_conc=100, c=0.001, k=0.0001) # when k and c are equal things dont make sense
#animation.simulate()
#CorT="temp" CorT="conc" and CorT="both" to plot each respectively against time
#animation.graph(CorT="both")

class Combined_linear_increasing:

    def __init__(self, length=50, time=50, nodes=30, Q=0.1, D_T=0.1068, D_a=0.025, beta=0.0411, c=0.001, k=0.0001,
                 initial_temp=27, initial_conc=100, boundary_temp=200):

        # Variables
        self.length = length
        self.time = time
        self.nodes = nodes
        self.Q = Q
        self.D_T = D_T
        self.D_a = D_a
        self.c = c
        self.k = k
        self.beta = beta
        self.initial_temp = initial_temp
        self.initial_conc = initial_conc
        self.boundary_temp = boundary_temp
        
        # Initialise step sizes 
        self.dx = self.length / self.nodes
        self.dy = self.length / self.nodes
        self.dt = self.dx**2 / 200  # smaller the better 

        # Initialise data matricies
        self.u = np.zeros((self.nodes, self.nodes)) + self.initial_temp
        self.a_values = np.zeros((self.nodes, self.nodes)) + self.initial_conc

        #Counters, flags and empty arrays 
        self.counter = 0
        self.blow_up_flag = False
        self.zero_conc = False
        self.break_counter = 0
        self.avg_temp = []
        self.avg_conc = []
        self.time_list = []

    def update_boundary_conditions(self, t):
        self.u[0, :] = self.boundary_temp * self.Q * t
        self.u[-1, :] = self.boundary_temp * self.Q * t
        self.u[:, 0] = self.boundary_temp * self.Q * t
        self.u[:, -1] = self.boundary_temp * self.Q * t

    def simulate(self):
        
        # Create subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

        # temp pcolormesh
        self.pcm_u = self.ax1.pcolormesh(self.u, cmap=plt.cm.jet, vmin=self.initial_temp, vmax=400) # max(self.boundary_temp_1, self.boundary_temp_2)
        self.colorbar_u = plt.colorbar(self.pcm_u, ax=self.ax1) 
        self.ax1.set_title("Temperature at t: {:.3f} [s]".format(0))

        # conc pcolormesh 
        self.pcm_a = self.ax2.pcolormesh(self.a_values, cmap=plt.cm.plasma, vmin=0, vmax=self.initial_conc)
        self.colorbar_a = plt.colorbar(self.pcm_a, ax=self.ax2)
        self.ax2.set_title("Concentration of A at t: {:.3f} [s]".format(0))

        self.avg_temp_marker = self.colorbar_u.ax.plot([0.5], [0], marker='_', color='white', markersize=25)
        self.avg_conc_marker = self.colorbar_a.ax.plot([0.5], [0], marker='_', color='black', markersize=25)

        while self.counter < self.time:
            
            w = self.u.copy()
            a_w = self.a_values.copy()

            for i in range(1, self.nodes-1):
                for j in range(1, self.nodes-1):
                    
                    # Fully-dimensionalised
                    dd_ax = (a_w[i-1, j] - 2*a_w[i, j] + a_w[i+1, j]) / self.dx**2
                    dd_ay = (a_w[i, j-1] - 2*a_w[i, j] + a_w[i, j+1]) / self.dy**2
                    self.a_values[i, j] = a_w[i, j] + self.dt * self.D_a *  (dd_ax + dd_ay) \
                        - self.dt * (self.k * a_w[i, j] * np.exp(self.beta* w[i, j]))

                    # Fully-dimensionalised
                    dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j]) / self.dx**2
                    dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1]) / self.dy**2
                    self.u[i, j] = w[i, j] + self.dt * self.D_T * (dd_ux + dd_uy) \
                        + self.dt * (self.c * a_w[i,j] * np.exp(self.beta * w[i, j]))

                    if self.u[i, j] > 3000 and not self.blow_up_flag:
                        self.pcm_u.set_cmap('hot')
                        self.pcm_u.set_array(self.u)
                        self.ax1.text(5, 15, f"Blow up at t={self.counter:.3f}", color="red", fontsize=20)
                        self.ax2.text(3, 15, f"Concentration zero at t={self.counter:.3f}", color="blue", fontsize=15)
                        self.blow_up_flag = True
                        self.u[:, 0] = 1e6
                        self.u[:, -1] = 1e6
                        self.break_counter = self.counter + 30*self.dt
                        break

                    if np.average(self.a_values[1:self.nodes-1, 1:self.nodes-1]) <= 0.1 and not self.zero_conc:
                        self.ax2.text(3, 15, f"Concentration zero at t={self.counter:.3f}", color="blue", fontsize=15)
                        self.zero_conc = True
                        self.break_counter = self.counter + 30*self.dt
                        break

            self.counter += self.dt
            self.update_boundary_conditions(self.counter)
            
            # avg temp for colorbar marker 
            avg_temp = np.mean(self.u[1:self.nodes-1, 1:self.nodes-1])
            self.avg_temp_marker[0].set_ydata(avg_temp) 

            # avg conc for colorbar marker 
            avg_conc = np.mean(self.a_values[1:self.nodes-1, 1:self.nodes-1])
            self.avg_conc_marker[0].set_ydata(avg_conc)

            if self.counter > self.break_counter and self.blow_up_flag is True:
                break

            if self.counter > self.break_counter and self.zero_conc is True:
                break
            
            self.pcm_u.set_array(self.u)
            self.pcm_a.set_array(self.a_values)
            self.ax2.set_title("Concerntration at t: {:.3f} [s] \nInitial Concentration of A: {:.1f}".format(self.counter, self.initial_conc))
            self.ax1.set_title("Temperature at t: {:.3f} [s] \nInitial Temperature {:.1f} C \nBoundary Temperatures {:.1f} C".format(self.counter, self.initial_temp, self.boundary_temp))
            plt.pause(0.1)

        plt.show()

    def graph(self, CorT=str):
        
        while self.counter < self.time:
            a_w = self.a_values.copy()
            w = self.u.copy()
            
            for i in range(1, self.nodes-1):
                for j in range(1, self.nodes-1):

                    # a_values following PDE
                    dd_ax = (a_w[i-1, j] - 2*a_w[i, j] + a_w[i+1, j]) / self.dx**2
                    dd_ay = (a_w[i, j-1] - 2*a_w[i, j] + a_w[i, j+1]) / self.dy**2
                    self.a_values[i, j] = a_w[i, j] + self.dt * self.D_a *  (dd_ax + dd_ay) \
                        - self.dt * (self.k * a_w[i, j] * np.exp(self.beta* w[i, j])) 

                    # u values following PDE 
                    dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j]) / self.dx**2
                    dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1]) / self.dy**2
                    self.u[i, j] = w[i, j] + self.dt * self.D_T * (dd_ux + dd_uy) \
                        + self.dt * (self.c * a_w[i,j] * np.exp(self.beta * w[i, j]))
                    
                    if self.u[i,j] > 1e6 and not self.blow_up_flag:
                        self.blow_up_flag = True
                        break

                    if np.mean(self.a_values[1:self.nodes-1, 1:self.nodes-1]) <= 0.01 and not self.zero_conc:
                        self.zero_conc = True
                        break


            # Creating a list for plotting temp vs time 
            avg_temp = np.average(self.u[1:self.nodes-1, 1:self.nodes-1])
            avg_conc = np.average(self.a_values[1:self.nodes-1, 1:self.nodes-1])
            self.avg_temp.append(avg_temp)
            self.avg_conc.append(avg_conc)
            self.time_list.append(self.counter)

            self.counter += self.dt

            self.update_boundary_conditions(self.counter)

            if self.blow_up_flag is True or self.zero_conc is True:
                break 

        if CorT == "both":
            plt.plot(self.time_list, self.avg_temp, color='red', label="Temperature")
            plt.plot(self.time_list, self.avg_conc, color='blue', label="Concentration")
            plt.legend(fontsize=15)
            plt.xlim(0, self.time)
            plt.ylim(-1, 200)
            plt.xlabel("Time [s]", fontsize=15)
            plt.ylabel("Avg temperature and Conc", fontsize=15)
            plt.title("Temperature and Conc on same graph against time \nInitial Temperature {:.1f} C\nBoundary Temperature {:.1f} C \nInitial Concentration {:.1f} mol/m^3".format(self.initial_temp, self.boundary_temp, self.initial_conc), fontsize=15)
            plt.show()

animation_combined = Combined_linear_increasing(time=40, Q=0.1)
animation_combined.simulate()
animation_combined.graph("both")
