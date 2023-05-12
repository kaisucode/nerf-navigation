import numpy as np
import hnswlib
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

np.random.seed(0)

# sys.path.insert(0, "/home/lishuo_p14s/nerf-navigation/NeRF/ngp_pl")  # TODO
sys.path.insert(0, "/home/rahulsajnani/Education/Brown/1_sem2/52-O/project/nerf-navigation/NeRF/ngp_pl")  # TODO

print(sys.path)
# sys.path.append("/home/lishuo_p14s/nerf-navigation/NeRF/ngp_pl")  # TODO
from odom_to_ngp import odom_to_nerf, get_odom_to_nerf_matrix
from get_density import ngp_model
import torch

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
    def __init__(self, n, start, height, goal, map_model_checkpoint_path, rotation_o2n, offset_o2n, scale):
        self.path_found = False
        # robot info
        self.robot_height = height
        self.robot_config_radius = 0.9
        self.collision_point_could_density_threshold = 190/255
        self.orientation = np.array([1,0,0,0])

        # nerf info
        self.rotation_o2n = rotation_o2n
        self.offset_o2n = offset_o2n
        # self.scale = scale
        self.map_model = ngp_model(scale, map_model_checkpoint_path)

        # map info
        self.h = 3.6
        self.w = 6
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
        self.r = min(50*(np.log(self.n)/self.n)**(1/self.dim), 3)

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
            # print("sample once")
            sample_x = np.random.uniform(0, self.w, 1)
            sample_y = np.random.uniform(-self.h / 2, self.h / 2, 1)
            sample = np.append(sample_x, sample_y)
            # sample = np.array([2.6, -0.4])            
            collision = self.is_in_collision(sample)
            # print(collision)
        return sample

    def steer(self, x1, x2):
        # returns the optimal control trajectory of traveling from x1 to x2 # and the cost
        cost = np.linalg.norm(x1 - x2)
        return cost

    def coordinate_odom_to_comap(self, coordinates):
        # coordinate [n,3]
        cur_position = coordinates
        cur_orientation = np.repeat(self.orientation[np.newaxis, :], coordinates.shape[0], axis=0)
        poses = odom_to_nerf(None, cur_position, cur_orientation, self.rotation_o2n, self.offset_o2n, to_ngp=True)
        
        return poses[:, :3, -1]
        
        # return coordinates_colmap[..., 0]  # [n,3]

    def mapping(self, coordinates):
        comap_coordinates = self.coordinate_odom_to_comap(coordinates)  # [n,3]
        # move the data to gpu
        comap_coordinates_gpu = torch.tensor(comap_coordinates).cuda()
        num_points = coordinates.shape[0]
        # print("comap_coordinates_gpu", comap_coordinates_gpu.shape)
        
        with torch.no_grad():
            densities = self.map_model.get_density(comap_coordinates_gpu)
        
        return densities  # [n,1]

    def is_in_collision(self, x, num_point_cloud=100):  # TODO modify this function
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

        # filter out the points under ground.
        ground_level = 0
        point_cloud_filter = []
        filtered_out_row = point_could_coordinates[:, 2] < ground_level
        for i in range(point_could_coordinates.shape[0]):
            if filtered_out_row[i] == False:
                point_cloud_filter.append(point_could_coordinates[i])
        point_could_coordinates = np.array(point_cloud_filter) # TODO check this dimension

        
        # print(point_could_coordinates)
        
        # query the map to get the density
        # densities_numpy = self.mapping(point_could_coordinates)[..., None].cpu().detach().numpy()
        # print("max of density:", np.max(densities_numpy))
        # print("min of density:", np.min(densities_numpy))
        densities = 1.0 - np.exp(-self.mapping(point_could_coordinates)[..., None].cpu().detach().numpy())  # [n, 1]
        # print(densities.shape)
        # assert(densities.shape == (num_point_cloud, 1))

        # print("start collision checking")
        
        # judge if the point is a valid sample
        avg_density = np.sum(densities) / num_point_cloud
        # print(avg_density)
        if avg_density > self.collision_point_could_density_threshold:
            return True
        else:
            return False

    def path_collision(self, x1, x2):
        # return true if the path is collision with the obstacle
        p = x1; r = x2-x1;

        # use the sampling method to decide if path is collide with mapping
        collision_check_distance = 0.3
        point_distance = np.linalg.norm(r)
        num_checks = int(np.ceil(point_distance // collision_check_distance))
        xx = np.linspace(x1[0], x2[0], num_checks+2)[1:-1]
        yy = np.linspace(x1[1], x2[1], num_checks+2)[1:-1]
        # print("xx shape", xx.shape)
        
        for x, y in zip(xx, yy):
            collide = self.is_in_collision(np.append(x, y))
            if collide:
                return True
        # print("path collision check finished")lishuo_p14s
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
        
        # comput the distance to the goal
        goal_diff_mat = np.array(self.V_location)[in_circle_idx] - self.goal
        goal_diff_vect = np.linalg.norm(goal_diff_mat, axis=1)
        argsort = np.argsort(goal_diff_vect)
        
        num_neighbor = np.sum(in_circle_idx)
        
        # goal_diff_mat = x - self.goal
        
        count = 0
        for idx, judge in enumerate(in_circle_idx):
            if judge:
                if argsort[count] < num_neighbor/2:
                    near_set.append(self.V_node[idx])
                count += 1

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
        X_near_goal = self.findNearSet(self.goal, 0.4)
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
        # rect1 = patches.Rectangle((4, -4), 1, 13, linewidth=1,facecolor="y", edgecolor='y', lw=2)
        # rect2 = patches.Rectangle((3.4, -4.6), 2.2, 14.2, linewidth=1, facecolor="none", edgecolor='y', lw=1, ls='--')
        # rect3 = patches.Rectangle((-6, -5), 6, 1, linewidth=1,facecolor="y", edgecolor='y', lw=2)
        # rect4 = patches.Rectangle((-6.6, -5.6), 7.2, 2.2, linewidth=1, facecolor="none", edgecolor='y', lw=1, ls='--')
        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)
        # ax.add_patch(rect4)

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
        return np.flip(path_mat, 0)

    def run(self):
        sample_iter = 0
        while (not self.path_found) or (sample_iter < self.n):
            if sample_iter % 10 == 0:
                print("sample points: ", sample_iter, "/", self.n) 
        # for i in tqdm.tqdm(range(self.n)):
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
    start = np.array([1.2, 0.2])
    height = 1
    goal = np.array([5, 0.2])
    
    parent_dir = "../../../../spot_data/" # TODO
    mapping_path = "../../../../spot_data/ckpts/spot_online/Spot/1_slim.ckpt" # TODO
    
    # parent_dir = "../../../../spot_data_best/spot_data/" # TODO
    # mapping_path = "../../../../spot_data_best/spot_data/ckpts/spot_online/Spot/2_slim.ckpt" # TODO
    
    colmap_scale = 0.5
    ts = np.load(os.path.join(parent_dir, "arr_2.npy"))
    qs = np.load(os.path.join(parent_dir, "arr_3.npy"))
    o2n, offset = get_odom_to_nerf_matrix(parent_dir, ts, qs, colmap_scale)
    
    rrt_star = RRTStar(80, start, height, goal, mapping_path, o2n, offset, scale=colmap_scale)
    map_model_checkpoint = ""

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
