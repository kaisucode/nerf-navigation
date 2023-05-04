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
    def __init__(self, n, start, goal):
        self.path_found = False
        # robot info
        self.robot_height = 1  # TODO fix this value.
        self.robot_config_radius = 0.7
        self.collision_point_could_density_threshold = 0.5

        # map info
        self.h = 20
        self.w = 20
        # self.sx = 0
        # self.sy = 0
        self.start = start

        self.x_min = start
        self.c_min = 0
        # RRT_star tree
        self.V_location = [start]
        self.V_parent_loc = []
        self.V_node = [Node(start, self.c_min, None)]
        self.E = []

        self.dim = 2
        self.n = n
        self.r = min(50*(np.log(self.n)/self.n)**(1/self.dim), 4)

        # self.joint_pts = [[np.array([3.4, 9.6]), np.array([3.4, -4.6])],
        #                   [np.array([3.4, 9.6]), np.array([5.6, 9.6])],
        #                   [np.array([3.4, -4.6]), np.array([5.6, -4.6])],
        #                   [np.array([5.6, 9.6]), np.array([5.6, -4.6])],
        #                   [np.array([-6.6, -5.6]), np.array([0.6, -5.6])],
        #                   [np.array([-6.6, -5.6]), np.array([-6.6, -3.4])],
        #                   [np.array([-6.6, -3.4]), np.array([0.6, -3.4])],
        #                   [np.array([0.6, -5.6]), np.array([0.6, -3.4])]]
        self.joint_pts = []

        self.lines = len(self.joint_pts)


        self.pts_idx = 0
        # Declaring index
        self.p = hnswlib.Index(space='l2', dim=self.dim)  # possible options are l2, cosine or ip

        # Initing index - the maximum number of elements should be known beforehand
        self.p.init_index(max_elements=10*self.n+1, ef_construction=200, M=16)

        # Controlling the recall by setting ef:
        self.p.set_ef(50)  # ef should always be > k

        # add init pt & increase the pts_idx
        self.p.add_items(start, self.pts_idx)
        self.pts_idx += 1

        # the goal pts
        # self.goal = np.array([8,8])
        self.goal = goal

        self.best_cost = []
    def sample(self):
        # returns a state x that is sampled uniformly randomly from the domain
        collision = True
        while collision:
            sample = np.random.uniform(0 - self.w/2, 0 + self.w/2, 2)
            collision = self.is_in_collision(sample)

        return sample

    def steer(self, x1, x2):
        # returns the optimal control trajectory of traveling from x1 to x2 # and the cost
        cost = np.linalg.norm(x1 - x2)
        return cost

    def mapping(self, coordinates):
        num_points = coordinates.shape[0]
        densities = np.zeros((num_points, 1))
        return densities

    def is_in_collision(self, x, num_point_cloud=50):  # TODO modify this function
        # returns True if state x of the robot is incollision with any of the
        # obstacles

        # given the robot position, sample points within a barrier volume
        aug_x = np.append(x, self.robot_height)  # [3, ]

        # generate the point-could within the volume
        u = np.random.normal(0, 1, num_point_cloud)
        v = np.random.normal(0, 1, num_point_cloud)
        w = np.random.normal(0, 1, num_point_cloud)
        r = self.robot_config_radius * np.random.rand(num_point_cloud) ** (1/3)
        norm = (u*u + v*v + w*w)**(1/2)
        point_could_coordinates = aug_x[np.newaxis, :] + r[:, np.newaxis] / (norm[:, np.newaxis]) * np.stack((u, v, w)).transpose()  # [n, 3]

        # query the map to get the density
        densities = self.mapping(point_could_coordinates)  # [n, 1]
        assert(densities.shape == (num_point_cloud, 1))

        # judge if the point is a valid sample
        avg_density = np.sum(densities) / num_point_cloud
        if avg_density > self.collision_point_could_density_threshold:
            return True
        else:
            return False

    def path_collision(self, x1, x2):
        # return true if the path is collision with the obstacle
        p = x1; r = x2-x1;

        # use the sampling method to decide if path is collide with mapping
        collision_check_distance = 0.5
        point_distance = np.linalg.norm(r)
        num_checks = int(np.ceil(point_distance // collision_check_distance))
        xx = np.linspace(x1[0], x2[0], num_checks+2)[1:-1]
        yy = np.linspace(x1[1], x2[1], num_checks+2)[1:-1]

        for x, y in zip(xx, yy):
            collide = self.is_in_collision(np.append(x, y))
            if collide:
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
            self.path_found = True
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
        plt.scatter(self.start[0], self.start[1], s=120, color="b", alpha=0.5)
        plt.scatter(self.goal[0], self.goal[1], s=100, color="g", alpha=0.5)

        plt.title("map")
        plt.show()

        plt.plot(self.best_cost)
        plt.title("cost v.s. iteration")
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()
        return path_mat

    def run(self):
        sample_iter = 0
        while (not self.path_found) or (sample_iter < self.n):
            x_rand_loc = self.sample()
            self.x_nearest = self.nearest(x_rand_loc)

            # judge if x_nearest could connect to x_rand
            if not self.path_collision(self.x_nearest.location, x_rand_loc):
                # find the near pts set in the circle
                self.X_near = self.findNearSet(x_rand_loc, self.r)

                # connect the x_rand with the tree
                self.connect(x_rand_loc)
                # rewire the tree
                self.rewire()

                self.getBestPath()

            sample_iter += 1

if __name__ == '__main__':
    start = np.array([7, -3])
    goal = np.array([8, 8])
    rrt_star = RRTStar(100, start, goal)

    rrt_star.run()
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
