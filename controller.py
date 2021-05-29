import numpy as np

from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import casadi as ca
import numpy as np

def Fast_Catch(a_state,d_state):
    is_poistive = 1
    distance = np.sqrt(np.sum(np.square(a_state[:2] - d_state[:2])))
    dx =a_state[0] - d_state[0]
    dy =a_state[1] - d_state[1]
    theta_e = np.arctan2(dy,dx) - d_state[2]
    # print(np.arctan(dy/dx))
    # print(d_state[2])
    # a_state[2] - d_state[2]
    if(theta_e>np.pi):
        theta_e -= 2*np.pi
    elif (theta_e<-np.pi):
        theta_e += 2*np.pi
    # print(theta_e)
    
    if(theta_e>np.pi/2 or theta_e<-np.pi/2):
        is_poistive = -1

    u = 1.5*distance*is_poistive
    w = theta_e*is_poistive
    u = np.clip(u, -0.5, 0.5)
    w = np.clip(w, -2, 2)
    return np.array([u,w])

def Fast_controller_defence(a_state,d_state):
    is_poistive = 1
    g = a_state.copy()
    g[0] = 0.5*g[0] + 0.5*1
    g[1] = 0.5*g[1] + 0.5*1
    distance = np.sqrt(np.sum(np.square(g[:2] - d_state[:2])))
    dx =g[0] - d_state[0]
    dy =g[1] - d_state[1]
    theta_e = np.arctan2(dy,dx) - d_state[2]
    # print(np.arctan(dy/dx))
    # print(d_state[2])
    # a_state[2] - d_state[2]
    if(theta_e>np.pi):
        theta_e -= 2*np.pi
    elif (theta_e<-np.pi):
        theta_e += 2*np.pi
    # print(theta_e)
    
    if(theta_e>np.pi/2 or theta_e<-np.pi/2):
        is_poistive = -1

    u = 1*distance*is_poistive
    w = theta_e*is_poistive
    u = np.clip(u, -0.4, 0.4)
    w = np.clip(w, -2, 2)
    return np.array([u,w])

def MPC_controller_defence(a_state,d_state):
    T = 0.05
    N = 20
    v_max = 0.5
    omega_max = 2

    opti = ca.Opti()
    # control variables, linear velocity v and angle velocity omega
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]


    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    opt_x0 = opti.parameter(3)
    opt_xs = opti.parameter(3)
    
    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])
    f_np = lambda x_, u_: np.array([u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)
        # opti.subject_to(opt_states[i+1, 1]<=10)
        

    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    W_slack = np.array([[100]])
    goal = np.array([[1.0, 0.0, 0.0]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - opt_xs.T), Q, (opt_states[i, :]- opt_xs.T).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])

    opti.minimize(obj)
    opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(opti.bounded(-2.0, y, 2.0))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))    
    # opts_setting = {'ipopt.max_iter':100}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    opti.solver('ipopt', opts_setting)
    final_state = a_state.copy()
    final_state[0] = (final_state[0]+1.0)/2
    
    opti.set_value(opt_xs, final_state)

    opti.set_value(opt_x0, d_state)
    sol = opti.solve()
    u_res = sol.value(opt_controls)
    return u_res[0, :]

def MPC_controller(a_state,d_state):
    T = 0.05
    N = 10
    v_max = 0.5
    omega_max = 2

    opti = ca.Opti()
    # control variables, linear velocity v and angle velocity omega
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]
    d_slack = opti.variable(N, 1)

    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    opt_x0 = opti.parameter(3)
    opt_xs = opti.parameter(3)
    
    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])
    f_np = lambda x_, u_: np.array([u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)
        # opti.subject_to(opt_states[i+1, 1]<=10)
        distance_constraints_ = (opt_states[i, 0].T - d_state[0]) ** 2 + (opt_states[i, 1].T - d_state[1]) ** 2 
        opti.subject_to(distance_constraints_ >= 0.12 + d_slack[i, :])

    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    Q2 = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.5, 0.0], [0.0, 0.001]])
    W_slack = np.array([[1000]])
    goal = np.array([[1.2, 1.2, 0.0]])
    #### cost function
    obj = 0 #### cost
    d_ = np.array([[d_state[0],d_state[1],d_state[2]]])
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - goal), Q, (opt_states[i, :]- goal).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) + ca.mtimes([d_slack[i, :], W_slack, d_slack[i, :].T])

    opti.minimize(obj)
    opti.subject_to(opti.bounded(-1.45, x, 1.45))
    opti.subject_to(opti.bounded(-1.45, y, 1.45))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))    
    opts_setting = {'ipopt.max_iter':500, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt',opts_setting)
    final_state = np.array([1.5, 0, 0.0])
    opti.set_value(opt_xs, final_state)
    opti.set_value(opt_x0, a_state)
    sol = opti.solve()
    u_res = sol.value(opt_controls)
    slack = sol.value(d_slack)
    # print(slack)
    return u_res[0, :]





def CLF_CBF_controller(a_state,d_state):
    

    gamma = 1
    x_state = np.array([[a_state[0]], [a_state[1]], [a_state[2]]])

    P_goal = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    P_obstacle = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 0]])

    # P_constrain = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]])

    C_goal = np.array([[1.5], [0], [0]])
    C_obstacle = np.array([[d_state[0]], [d_state[1]], [d_state[2]]])

    # print(C_obstacle)

    C_constrain = np.array([[0], [1], [0]])

    V_goal = np.dot(np.dot(np.transpose(x_state - C_goal), P_goal), x_state - C_goal)  ##value
    H_obstacle = np.dot(np.dot(np.transpose(x_state - C_obstacle), P_obstacle), x_state - C_obstacle)
    H_constrain = -np.dot(np.dot(np.transpose(x_state - C_constrain), P_obstacle), x_state - C_constrain)

    # Quadratic program declarations
    A_goal = np.dot(2 * np.transpose(x_state - C_goal), P_goal) 		# Constraint for
    B_goal = - gamma * V_goal	        # goal region
    G =  np.array([[np.cos(a_state[2]), 0], [np.sin(a_state[2]), 0], [0, 1]])
    A_goal = A_goal@G
    A_goal = np.append(A_goal,-1)
	
    A_obs = np.dot(-2.0 * np.transpose(x_state - C_obstacle), P_obstacle) 	# Constraint for
    B_obs = gamma * (H_obstacle)						# obstacle
    A_obs = A_obs@G
    A_obs = np.append(A_obs,0)
    # print(A_goal)
            
    # A_obs = np.dot(-2.0 * np.transpose(x_state - C_obstacle1), P_obstacle1) 	# Constraint for
    # B_obs1 = gamma * (h_obstacle1)**5						# obstacle
    # A_obs1 = np.append(A_obs1, -1)
    A = np.vstack((A_goal,A_obs))
    B = np.vstack((B_goal,B_obs))

    # Quadratic program solver
    H = matrix(np.array([[1, 0, 0], [0, 0.5, 0], [0,0,100]]), tc='d')
    f = matrix(np.array([[0], [0], [0]]), tc='d')
    u = qp(H, f, matrix(A), matrix(B))['x']
    return u
    # Obtain unicycle velocities using Diffeomorphism technique
    # l = 0.3
    # R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # L = np.array([[1, 0], [0, 1/l]])
    # u_turtlebot = np.dot(np.dot(np.transpose(R), L), np.array(u_si)[:2])

    # # Publish the velocities to Turtlebot
    # vel_msg.linear.x = u_turtlebot[0] * np.cos(phi)
    # vel_msg.linear.y = u_turtlebot[0] * np.sin(phi)
    # vel_msg.linear.z = 0

    # vel_msg.angular.x = 0
    # vel_msg.angular.y = 0
    # vel_msg.angular.z = u_turtlebot[1]

    # velocity_publisher.publish(vel_msg)
if __name__ == '__main__':
    states = np.array([[-1,0,0],[0.5,0,-np.pi]])
    u = MPC_controller(states[0],states[1])
    print(u)