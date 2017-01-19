from collections import namedtuple
import numpy as np

MAP_1 = """
###.*###
###s.###
###..###
*.....w.
.e.....*
###..###
###.n###
###*.###
"""

MAP_2 = """
####*###
####.###
####.###
*.....w.
.e.....*
####.###
####n###
####.###
"""

MAP_3 = """
###.*###
###s.###
###..###
*....###
.e...###
###..###
###.n###
###*.###
"""

MAP_4 = """
##.*####
##s.####
##..####
*.....w.
.e.....*
####..##
####.n##
####*.##
"""

MAP_5 = """
##***###
##...###
##...###
##...###
##...###
##...###
##nnn###
##...###
"""

MAPS = [MAP_1, MAP_2, MAP_3, MAP_4, MAP_5]

MAP_SHAPE = (8, 8)

DIRS = {"n": 0, "e": 1, "s": 2, "w": 3}

Car = namedtuple("Car", ["pos", "dir", "goal", "done"])

N_FEATURES = (
    len(MAPS)
    + MAP_SHAPE[0] * MAP_SHAPE[1] * 2
    + len(DIRS))


class DriveTask(object):
    def __init__(self):
        self.random = np.random.RandomState(0)
        self.n_agents = 2
        self.symmetric = True
        self.n_actions = (4, 4)
        self.n_features = N_FEATURES
        self.max_desc_len = 1
        self.n_vocab = 1
        self.vocab = {"UNK": 0}
        self.reverse_vocab = {0: "UNK"}

    def get_instance(self, _):
        map_id = self.random.randint(len(MAPS))
        map_str = MAPS[map_id]
        template = map_str.split("\n")[1:-1]
        rows = len(template)
        cols = len(template[0])
        start_indices = [
                (r, c) for r in range(rows) for c in range(cols)
                if template[r][c] in ("n", "e", "s", "w")]
        goal_indices = [
                (r, c) for r in range(rows) for c in range(cols)
                if template[r][c] == "*"]

        agent1_start = start_indices[self.random.randint(len(start_indices))]
        agent1_dir = DIRS[template[agent1_start[0]][agent1_start[1]]]
        start_indices.remove(agent1_start)
        agent2_start = start_indices[self.random.randint(len(start_indices))]
        agent2_dir = DIRS[template[agent2_start[0]][agent2_start[1]]]
        agent1_goal = goal_indices[self.random.randint(len(goal_indices))]
        agent2_goal = goal_indices[self.random.randint(len(goal_indices))]

        road = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                road[r, c] = 0 if template[r][c] == "#" else 1

        cars = [Car(agent1_start, agent1_dir, agent1_goal, False),
                Car(agent2_start, agent2_dir, agent2_goal, False)]

        return DriveState(map_id, road, cars)

    def distractors_for(self, state, n_samples):
        return [(state, 1)] * n_samples

    def visualize(self, state, agent):
        draw = [[None for _ in range(MAP_SHAPE[1])] for _ in range(MAP_SHAPE[0])]
        car = state.cars[agent]

        if car.done:
            return "<pre>done</pre>"

        for r in range(MAP_SHAPE[0]):
            for c in range(MAP_SHAPE[1]):
                if state.road[r, c]:
                    draw[r][c] = "."
                else:
                    draw[r][c] = " "

        r, c = car.pos
        r2, c2 = r, c
        draw[r][c] = "C"
        if car.dir == 0:
            r2 += 1
        elif car.dir == 1:
            c2 -= 1
        elif car.dir == 2:
            r2 -= 1
        elif car.dir == 3:
            c2 += 1
        if 0 <= r2 < state.road.shape[0] and 0 <= c2 < state.road.shape[1]:
            draw[r2][c2] = "C"

        rg, cg = car.goal
        draw[rg][cg] = "*"

        return "<pre>" + "\n".join(["".join(r) for r in draw]) + "</pre>" #\
            #"\n" + str(hash(tuple(state.obs()[agent]))) + "</pre>"

    def pp(self, indices):
        return " ".join([self.reverse_vocab[i] for i in indices])

class DriveState(object):
    def __init__(self, map_id, road, cars):
        self.map_id = map_id
        self.road = road
        self.cars = cars
        self.desc = [0]

    def obs(self):
        return tuple(self._obs(car) for car in self.cars)

    def _obs(self, car):
        map_features = np.zeros(len(MAPS))
        map_features[self.map_id] = 1
        if car.done:
            return np.zeros(N_FEATURES)
        goal = np.zeros(self.road.shape)
        pos = np.zeros(self.road.shape)
        direction = np.zeros((len(DIRS),))
        pos[car.pos] = 1
        goal[car.goal] = 1
        direction[car.dir] = 1
        return np.concatenate(
                (map_features, pos.ravel(), goal.ravel(), direction))

    def _move(self, car, action):
        nr, nc = car.pos
        ndir = car.dir
        turn = False
        if action == 0:
            pass
        elif action == 1:
            if car.dir == 0:
                nr -= 1
            elif car.dir == 1:
                nc += 1
            elif car.dir == 2:
                nr += 1
            elif car.dir == 3:
                nc -= 1
        elif action == 2:
            ndir -= 1
            turn = True
        elif action == 3:
            ndir += 1
            turn = True
        ndir %= 4

        if turn:
            if ndir == 0:
                nr -= 1
            elif ndir == 1:
                nc += 1
            elif ndir == 2:
                nr += 1
            elif ndir == 3:
                nc -= 1

        nr = min(nr, self.road.shape[0]-1)
        nc = min(nc, self.road.shape[1]-1)
        nr = max(nr, 0)
        nc = max(nc, 0)

        return Car((nr, nc), ndir, car.goal, car.done)

    def step(self, actions):
        assert len(actions) == len(self.cars)
        ncars = [self._move(c, a) for c, a in zip(self.cars, actions)]
        occupied = np.zeros(self.road.shape)
        heads = np.zeros(self.road.shape)
        tails = np.zeros(self.road.shape)
        for car in ncars:
            if car.done:
                continue
            pos = car.pos
            r2, c2 = car.pos
            if car.dir == 0:
                r2 += 1
            elif car.dir == 1:
                c2 -= 1
            elif car.dir == 2:
                r2 -= 1
            elif car.dir == 3:
                c2 += 1
            occupied[pos] += 1
            heads[pos] += 1
            if 0 <= r2 < self.road.shape[0] and 0 <= c2 < self.road.shape[1]:
                occupied[r2, c2] += 1
                tails[r2, c2] += 1

        off = 0
        crash = False
        draw = [[None for _ in range(MAP_SHAPE[1])] for _ in range(MAP_SHAPE[0])]
        for r in range(self.road.shape[0]):
            for c in range(self.road.shape[1]):
                draw[r][c] = "." if self.road[r][c] else " "
                for car in ncars:
                    draw[car.goal[0]][car.goal[1]] = "G"
                if heads[r, c] > 0:
                    draw[r][c] = "#"
                if tails[r, c] > 0:
                    draw[r][c] = "O"
                if occupied[r, c] > 0 and self.road[r, c] == 0:
                    off += 1
                if occupied[r, c] > 1:
                    crash = True

        #reverse = len([a for a in actions if a == 1])

        final_cars = []
        n_success = 0
        for car in ncars:
            if car.pos == car.goal and not car.done:
                n_success += 1
                final_cars.append(car._replace(done=True))
            else:
                final_cars.append(car)

        reward = 0
        stop = False
        if all(car.done for car in final_cars):
            stop = True
        if crash:
            reward -= 1
            stop = True
        reward -= 0.01
        reward -= 0.1 * off
        #reward -= 0.05 * reverse
        reward += 0.5 * n_success

        #print "\n".join(["".join(row) for row in draw])
        #print reward
        #print

        return DriveState(self.map_id, self.road, final_cars), reward, stop