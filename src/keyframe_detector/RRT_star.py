import numpy as np
import hnswlib
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class Node:
    def __init__(self, location, cost = np.inf, parent = None, child = None):
        self.location = location
        self.cost = cost
        self.parent = parent
        self.child = child
    def setCost(self, cost):
        self.cost = cost
    def setParent(self, parent):
        self.parent = parent
    def setChild(self, child):
        self.child = child


class RRTStar:
    def __init__(self, n):
        # map info
        self.h = 20
        self.w = 20
        self.obs = [[],[]]
        self.sx = 0
        self.sy = 0

        self.x_min = np.array([self.sx,self.sy])
        self.c_min = 0
        # RRT_star tree
        self.V_location = [np.array([self.sx, self.sy])]
        self.V_parent_loc = []
        self.V_node = [Node(np.array([self.sx, self.sy]), self.c_min, None)]
        self.E = []

        self.dim = 2
        self.n = n
        self.r = min(50*(np.log(self.n)/self.n)**(1/self.dim), 4)

        self.joint_pts = [[np.array([3.4, 9.6]), np.array([3.4, -4.6])],
                          [np.array([3.4, 9.6]), np.array([5.6, 9.6])],
                          [np.array([3.4, -4.6]), np.array([5.6, -4.6])],
                          [np.array([5.6, 9.6]), np.array([5.6, -4.6])],
                          [np.array([-6.6, -5.6]), np.array([0.6, -5.6])],
                          [np.array([-6.6, -5.6]), np.array([-6.6, -3.4])],
                          [np.array([-6.6, -3.4]), np.array([0.6, -3.4])],
                          [np.array([0.6, -5.6]), np.array([0.6, -3.4])]]
        self.lines = len(self.joint_pts)


        self.pts_idx = 0
        # Declaring index
        self.p = hnswlib.Index(space='l2', dim=self.dim)  # possible options are l2, cosine or ip

        # Initing index - the maximum number of elements should be known beforehand
        self.p.init_index(max_elements=self.n+1, ef_construction=200, M=16)

        # Controlling the recall by setting ef:
        self.p.set_ef(50)  # ef should always be > k

        # add init pt & increase the pts_idx
        self.p.add_items(np.array([self.sx,self.sy]), self.pts_idx)
        self.pts_idx += 1

        # the goal pts
        self.goal = np.array([8,8])

        self.best_cost = []
    def sample(self):
        # returns a state x that is sampled uniformly randomly from the domain
        collision = True
        while collision:
            sample = np.random.uniform(self.sx - self.w/2, self.sx + self.w/2, 2)
            collision = self.is_in_collision(sample)

        return sample

    def steer(self, x1, x2):
        # returns the optimal control trajectory of traveling from x1 to x2 # and the cost
        cost = np.linalg.norm(x1 - x2)
        return cost


    def is_in_collision(self, x):
        # returns True if state x of the robot is incollision with any of the
        # obstacles

        # two obstacle
        if 0.6 >= x[0] >= -6.6 and -3.4 >= x[1] >= -5.6:
            return True
        if 5.6 >= x[0] >= 3.4 and 9.6 >= x[1] >= -4.6:
            return True

        return False

    def path_collision(self, x1, x2):
        # return true if the path is collision with the obstacle
        p = x1; r = x2-x1;

        for idx, element in enumerate(self.joint_pts):
            q = element[0]
            s = element[1] - element[0]

            rs_cross = self.cross(r, s)
            t = self.cross((q-p), s)/rs_cross
            u = self.cross((q-p), r)/rs_cross

            if rs_cross != 0 and 1 >= t >= 0 and 1 >= u >= 0:
                return True

        return False

    def cross(self,x1, x2):
        return x1[0]*x2[1] - x1[1]*x2[0]

    def nearest(self, x):
        # finds a node in the tree that is closest to the state x (closest in what
        # metric?)
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        labels, distances = self.p.knn_query(x, k=1)

        return self.V_node[labels.item()]

    def findNearSet(self, x, radius):
        near_set = []
        diff_mat = self.V_location - x
        diff_vect = np.linalg.norm(diff_mat, axis=1)
        in_circle_idx = diff_vect < radius

        for idx, judge in enumerate(in_circle_idx):
            if judge:
                near_set.append(self.V_node[idx])

        return near_set

    def connect(self, x_rand_loc):
        # add state x to the tree
        # cost of state x = cost of the parent + cost of steer(parent, x)

        ## find the x_min and c_min
        self.x_min = self.x_nearest
        self.c_min = self.x_nearest.cost + self.steer(self.x_nearest.location, x_rand_loc)
        # go through all the x_near, find the shortest path and update x_min and c_min
        for x_near in self.X_near:
            x_near_path_cost = x_near.cost + self.steer(x_near.location, x_rand_loc)
            if (not self.path_collision(x_near.location, x_rand_loc)) and x_near_path_cost < self.c_min:
                self.x_min = x_near
                self.c_min = x_near_path_cost

        # add edge (x_min, x_rand) to the tree and
        # update the node with the cost and parent

        x_new = Node(x_rand_loc, self.c_min, self.x_min)
        self.x_min.child = x_new
        self.V_node.append(x_new)
        self.V_location.append(x_new.location)
        self.V_parent_loc.append(x_new.parent.location)
        self.p.add_items(x_new.location, self.pts_idx)
        self.pts_idx += 1
        pass

    def rewire(self):
        # here the new node has been added into the tree
        x_new = self.V_node[-1]
        # rewire all nodes in the tree within the O(gamma (log n/n)ˆ{1/d}} ball # near the state x, update the costs of all rewired neighbors
        for x_near in self.X_near:
            cost_new_near = x_new.cost + self.steer(x_new.location, x_near.location)
            if (not self.path_collision(x_near.location, x_new.location)) and cost_new_near < x_near.cost:
                x_near.parent = x_new
                # update the cost of near's child
                cur = x_near
                cur.cost = cost_new_near
                child_leaf = cur.child
                while child_leaf != None:
                    cur = cur.child
                    cur.cost = cur.cost + cost_new_near - x_near.cost
                    child_leaf = cur.child

        pass

    def getBestPath(self):
        clist = []
        X_near_goal = self.findNearSet(self.goal, 0.6)
        if len(X_near_goal) == 0:
            self.best_cost.append(np.inf)
        else:
            # add costs
            for element in X_near_goal:
                clist.append(element.cost)
            best_idx = np.argmin(clist)
            self.best_fin_pt = X_near_goal[best_idx]
            self.best_cost.append(self.best_fin_pt.cost)
        pass

    def plotAll(self):
        V_loc_mat = np.vstack(self.V_location)
        V_par_mat = np.vstack(self.V_parent_loc)
        # init the plot
        fig, ax = plt.subplots(1)
        # first plot the pts
        plt.scatter(V_loc_mat[:,0],V_loc_mat[:,1], s=0.2)

        ## plot the path
        # from the final pt
        cur = self.best_fin_pt
        # a list to store loc
        path = [cur.location]
        # indicator
        root = False
        while not root:  # while not the root
            cur = cur.parent
            if cur != None:
                path.append(cur.location)
            else:
                root = True
        path_mat = np.vstack(path)
        ax.plot(path_mat[:, 0], path_mat[:, 1],color="r")
        #plot the tree
        x_coor = np.vstack((V_loc_mat[1:,0], V_par_mat[:,0]))
        y_coor = np.vstack((V_loc_mat[1:,1], V_par_mat[:,1]))
        ax.plot(x_coor, y_coor,lw=0.1,color='g')
        # plot the obstacle
        # obstacle1 = np.asarray([[4, -4], [5, -4], [5, 9], [4, 9], [4, -4]])
        # obstacle2 = np.asarray([[-6, -4], [0, -4], [0, -5], [-6, -5], [-6, -4]])
        # plt.plot(obstacle1[:, 0], obstacle1[:, 1], 'b')
        # plt.plot(obstacle2[:, 0], obstacle2[:, 1], 'b')
        rect1 = patches.Rectangle((4, -4), 1, 13, linewidth=1,facecolor="y", edgecolor='y', lw=2)
        rect2 = patches.Rectangle((3.4, -4.6), 2.2, 14.2, linewidth=1, facecolor="none", edgecolor='y', lw=1, ls='--')
        rect3 = patches.Rectangle((-6, -5), 6, 1, linewidth=1,facecolor="y", edgecolor='y', lw=2)
        rect4 = patches.Rectangle((-6.6, -5.6), 7.2, 2.2, linewidth=1, facecolor="none", edgecolor='y', lw=1, ls='--')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)

        # plot the start point and end point
        plt.scatter(0,0,s=120,color="b",alpha=0.5)
        plt.scatter(8,8,s=100,color="g",alpha=0.5)

        plt.title("map")
        plt.show()

        plt.plot(self.best_cost)
        plt.title("cost v.s. iteration")
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()
        return path_mat
if __name__ == '__main__':
    rrt_star = RRTStar(10000)
    for i in tqdm.tqdm(range(rrt_star.n)):
        # cur_idx_V = i+1
        x_rand_loc = rrt_star.sample()
        rrt_star.x_nearest = rrt_star.nearest(x_rand_loc)

        # judge if x_nearest could connect to x_rand
        if not rrt_star.path_collision(rrt_star.x_nearest.location, x_rand_loc):
            # find the near pts set in the circle
            rrt_star.X_near = rrt_star.findNearSet(x_rand_loc, rrt_star.r)

            # connect the x_rand with the tree
            rrt_star.connect(x_rand_loc)
            # rewire the tree
            rrt_star.rewire()

            rrt_star.getBestPath()
    # V_loc_mat = np.vstack(rrt_star.V_location)
    # V_par_mat = np.vstack(rrt_star.V_parent_loc)
    # plt.scatter(V_loc_mat[:, 0], V_loc_mat[:, 1], s=0.2)
    # x_coor = np.vstack((V_loc_mat[1:, 0], V_par_mat[:, 0]))
    # y_coor = np.vstack((V_loc_mat[1:, 1], V_par_mat[:, 1]))
    # plt.plot(x_coor, y_coor,lw=0.1,color='g')
    # plt.show()
    rrt_star.getBestPath()
    rrt_star.best_cost[-1]
    path_mat = rrt_star.plotAll()
    print(path_mat)
    print()
