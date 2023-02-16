# Solution of CVRP by PSO
import math
import random
import numpy as np
import matplotlib.pyplot as plt


# calculate the distance between these customers
def distance(customer):
    distance_matrix = np.zeros((len(customer), len(customer)))
    for i in range(len(customer)):
        for j in range(len(customer)):
            distance_matrix[i][j] = math.sqrt(
                (customer[i][0] - customer[j][0]) ** 2
                + (customer[i][1] - customer[j][1]) ** 2
            )

    return distance_matrix


# initialize the population
def init_population(customer, distance_matrix, population_size):
    population = []
    # choose the first customer randomly
    for i in range(population_size):
        route = [random.randint(1, len(customer) - 1)]
        # choose the other customers greedily
        while len(route) < len(customer) - 1:
            # choose the nearest customer except the depot
            min_distance = float("inf")
            for j in range(1, len(customer)):
                if j not in route:
                    if min_distance > distance_matrix[route[-1]][j]:
                        min_distance = distance_matrix[route[-1]][j]
                        next_customer = j
            route.append(next_customer)
        population.append(route)

    return population


# assign the route to the vehicle
def assign_route(
    route,
    vehicle_capacity,
    demand,
    distance_matrix,
    vehicle_max_distance,
    cost_start,
    cost_per_distance,
):
    # each vehicle starts from the depot
    vehicle_route = [[0]]
    # the load of each vehicle
    vehicle_load = [0]
    # the distance of each vehicle
    vehicle_distance = [0]
    # the cost of each vehicle
    vehicle_cost = [0]
    # the number of vehicles
    vehicle_num = 0
    # assign the route to the vehicle
    for i in range(len(route)):
        # if the vehicle is full, return to the depot and start a new vehicle
        if vehicle_load[vehicle_num] + demand[route[i]] > vehicle_capacity:
            vehicle_route[vehicle_num].append(0)
            vehicle_distance[vehicle_num] += distance_matrix[route[i - 1]][0]
            vehicle_cost[vehicle_num] += (
                cost_start + cost_per_distance * vehicle_distance[vehicle_num]
            )
            vehicle_route.append([0])
            vehicle_load.append(0)
            vehicle_distance.append(0)
            vehicle_cost.append(0)
            vehicle_num += 1
        # if the vehicle is too far away from the depot, return to the depot and start a new vehicle
        elif (
            vehicle_distance[vehicle_num]
            + distance_matrix[route[i - 1]][route[i]]
            + distance_matrix[route[i]][0]
            > vehicle_max_distance
        ):
            vehicle_route[vehicle_num].append(0)
            vehicle_distance[vehicle_num] += distance_matrix[route[i - 1]][0]
            vehicle_cost[vehicle_num] += (
                cost_start + cost_per_distance * vehicle_distance[vehicle_num]
            )
            vehicle_route.append([0])
            vehicle_load.append(0)
            vehicle_distance.append(0)
            vehicle_cost.append(0)
            vehicle_num += 1
        # if the vehicle gets the last customer, return to the depot
        elif i == len(route) - 1:
            vehicle_route[vehicle_num].append(route[i])
            vehicle_route[vehicle_num].append(0)
            vehicle_load[vehicle_num] += demand[route[i]]
            vehicle_distance[vehicle_num] += distance_matrix[route[i - 1]][route[i]]
            vehicle_distance[vehicle_num] += distance_matrix[route[i]][0]
            vehicle_cost[vehicle_num] += (
                cost_start + cost_per_distance * vehicle_distance[vehicle_num]
            )
        # if the vehicle is not full, add the customer to the vehicle
        else:
            vehicle_route[vehicle_num].append(route[i])
            vehicle_load[vehicle_num] += demand[route[i]]
            vehicle_distance[vehicle_num] += distance_matrix[route[i - 1]][route[i]]

    # calculate the sum of the cost of all vehicles as the fitness of the route
    fitness = sum(vehicle_cost)

    return vehicle_route, fitness


# update each particle base on crossover
def crossover(particle, pbest, gbest, w, c1, c2):
    # order crossover
    child = []

    # parent1 is the particle itself
    parent1 = particle

    # w/(w+c1+c2) of probability to choose the particle itself
    # c1/(w+c1+c2) of probability to choose the pbest
    # c2/(w+c1+c2) of probability to choose the gbest
    if random.random() < w / (w + c1 + c2):
        parent2 = particle
    elif random.random() < (w + c1) / (w + c1 + c2):
        parent2 = pbest
    else:
        parent2 = gbest

    # choose the points randomly
    point1 = random.randint(0, len(parent1) - 1)
    point2 = random.randint(0, len(parent1) - 1)
    start_point = min(point1, point2)
    end_point = max(point1, point2)

    # add the customers between the points to the child
    for i in range(start_point, end_point + 1):
        child.append(parent1[i])

    # add the other customers to the child
    for i in range(len(parent2)):
        if parent2[i] not in child:
            child.append(parent2[i])

    return child


# draw the route
def draw_route(vehicle_route, customer):
    # draw the route of each vehicle
    for i in range(len(vehicle_route)):
        x = []
        y = []
        for j in range(len(vehicle_route[i])):
            x.append(customer[vehicle_route[i][j]][0])
            y.append(customer[vehicle_route[i][j]][1])
        plt.plot(x, y, marker="o")

    # draw the depot
    plt.plot(customer[0][0], customer[0][1], marker="s", color="red")

    plt.show()


def main():
    # parameters
    vehicle_capacity = 200
    vehicle_max_distance = 300
    cost_start = 30
    cost_per_distance = 1

    population_size = 50
    w = 0.2
    c1 = 0.4
    c2 = 0.4
    pbest_route = []
    pbest_fitness = float("inf")
    gbest_route = []
    gbest_fitness = float("inf")

    max_iteration = 1000
    best_fitness = []

    # read the data
    customer = [
        (50, 50),
        (96, 24),
        (40, 5),
        (49, 8),
        (13, 7),
        (29, 89),
        (48, 30),
        (84, 39),
        (14, 47),
        (2, 24),
        (3, 82),
        (65, 10),
        (98, 52),
        (84, 25),
        (41, 69),
        (1, 65),
        (51, 71),
        (75, 83),
        (29, 32),
        (83, 3),
        (50, 93),
        (80, 94),
        (5, 42),
        (62, 70),
        (31, 62),
        (19, 97),
        (91, 75),
        (27, 49),
        (23, 15),
        (20, 70),
        (85, 60),
        (98, 85),
    ]

    demand = [
        0,
        16,
        11,
        6,
        10,
        7,
        12,
        16,
        6,
        16,
        8,
        14,
        7,
        16,
        3,
        22,
        18,
        19,
        1,
        14,
        8,
        12,
        4,
        8,
        24,
        24,
        2,
        10,
        15,
        2,
        14,
        9,
    ]

    distance_matrix = distance(customer)

    # initialize the population
    population = init_population(customer, distance_matrix, population_size)

    # calculate the fitness of each particle
    fitness_list = []
    for i in range(population_size):
        vehicle_route, fitness_value = assign_route(
            population[i],
            vehicle_capacity,
            demand,
            distance_matrix,
            vehicle_capacity,
            cost_start,
            cost_per_distance,
        )

        fitness_list.append(fitness_value)

    # update the pbest and gbest
    for i in range(population_size):
        if fitness_list[i] < pbest_fitness:
            pbest_fitness = fitness_list[i]
            pbest_route = population[i]

        if fitness_list[i] < gbest_fitness:
            gbest_fitness = fitness_list[i]
            gbest_route = population[i]

    # update the best fitness
    best_fitness.append(gbest_fitness)

    # update the population
    for i in range(max_iteration):
        for j in range(population_size):
            child = crossover(population[j], pbest_route, gbest_route, w, c1, c2)

            vehicle_route, fitness_value = assign_route(
                child,
                vehicle_capacity,
                demand,
                distance_matrix,
                vehicle_capacity,
                cost_start,
                cost_per_distance,
            )
            # update the pbest and gbest
            if fitness_value < pbest_fitness:
                pbest_fitness = fitness_value
                pbest_route = child

            if fitness_value < gbest_fitness:
                gbest_fitness = fitness_value
                gbest_route = child

        # update the best fitness
        best_fitness.append(gbest_fitness)

    # draw the route
    vehicle_route, fitness_value = assign_route(
        gbest_route,
        vehicle_capacity,
        demand,
        distance_matrix,
        vehicle_capacity,
        cost_start,
        cost_per_distance,
    )
    draw_route(vehicle_route, customer)

    # draw the best fitness
    plt.plot(best_fitness)
    plt.show()


if __name__ == "__main__":
    main()
