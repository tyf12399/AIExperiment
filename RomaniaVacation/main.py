# solution for the "Romania Vacation" problem
import matplotlib.pyplot as plt


# draw
def draw_graph(graph, city_coordinates, path=None):
    plt.figure(figsize=(20, 20))
    for city in graph:
        for neighbor in graph[city]:
            x1, y1 = city_coordinates[city]
            x2, y2 = city_coordinates[neighbor]
            plt.plot([x1, x2], [y1, y2], "b")
    for city in city_coordinates:
        x, y = city_coordinates[city]
        plt.plot(x, y, "ro")
        plt.text(x, y, city, fontsize=10)
    if path:
        for i in range(len(path) - 1):
            x1, y1 = city_coordinates[path[i]]
            x2, y2 = city_coordinates[path[i + 1]]
            plt.plot([x1, x2], [y1, y2], "r")

    plt.show()


# Use BFS to find the shortest path
def bfs(graph, start, goal):
    explored = []
    queue = [[start]]

    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbors = graph[node]
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                if neighbor == goal:
                    return new_path
            explored.append(node)

    return None


# Use DFS to find the shortest path
def dfs(graph, start, goal):
    explored = []
    stack = [[start]]

    while stack:
        path = stack.pop(-1)
        node = path[-1]
        if node not in explored:
            neighbors = graph[node]
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)
                if neighbor == goal:
                    return new_path
            explored.append(node)

    return None


# Use A* algorithm to find the shortest path
def heuristic(city1, city2, graph):
    edges = graph[city1]
    for city in edges:
        if city == city2:
            return edges[city]
    return 999


def a_star(graph, start, goal):
    explored = []
    queue = [[start]]

    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbors = graph[node]
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                if neighbor == goal:
                    return new_path
            explored.append(node)
            queue.sort(key=lambda x: heuristic(x[-1], goal, graph))

    return None


# main
if __name__ == "__main__":
    # city data
    graph = {
        "Arad": {"Zerind": 75, "Sibiu": 140, "Timisoara": 118},
        "Bucharest": {"Urziceni": 85, "Giurgiu": 90, "Fagaras": 211, "Pitesti": 101},
        "Craiova": {"Drobeta": 120, "Rimnicu": 146, "Pitesti": 138},
        "Drobeta": {"Mehadia": 75, "Craiova": 120},
        "Eforie": {"Hirsova": 86},
        "Fagaras": {"Sibiu": 99, "Bucharest": 211},
        "Giurgiu": {"Bucharest": 90},
        "Hirsova": {"Urziceni": 98, "Eforie": 86},
        "Iasi": {"Neamt": 87, "Vaslui": 92},
        "Lugoj": {"Timisoara": 111, "Mehadia": 70},
        "Mehadia": {"Lugoj": 70, "Drobeta": 75},
        "Neamt": {"Iasi": 87},
        "Oradea": {"Zerind": 71, "Sibiu": 151},
        "Pitesti": {"Rimnicu": 97, "Craiova": 138, "Bucharest": 101},
        "Rimnicu": {"Sibiu": 80, "Pitesti": 97, "Craiova": 146},
        "Sibiu": {"Arad": 140, "Oradea": 151, "Fagaras": 99, "Rimnicu": 80},
        "Timisoara": {"Arad": 118, "Lugoj": 111},
        "Urziceni": {"Vaslui": 142, "Hirsova": 98, "Bucharest": 85},
        "Vaslui": {"Iasi": 92, "Urziceni": 142},
        "Zerind": {"Arad": 75, "Oradea": 71},
    }

    city_coordinates = {
        "Arad": (91, 492),
        "Bucharest": (400, 327),
        "Craiova": (253, 288),
        "Drobeta": (165, 299),
        "Eforie": (562, 293),
        "Fagaras": (305, 449),
        "Giurgiu": (375, 270),
        "Hirsova": (534, 350),
        "Iasi": (473, 506),
        "Lugoj": (165, 379),
        "Mehadia": (168, 339),
        "Neamt": (406, 537),
        "Oradea": (131, 571),
        "Pitesti": (320, 368),
        "Rimnicu": (233, 410),
        "Sibiu": (207, 457),
        "Timisoara": (94, 410),
        "Urziceni": (456, 350),
        "Vaslui": (509, 444),
        "Zerind": (108, 531),
    }

    path_bfs = bfs(graph, "Arad", "Bucharest")
    path_dfs = dfs(graph, "Arad", "Bucharest")
    path_a_star = a_star(graph, "Arad", "Bucharest")

    print("BFS: ", path_bfs)
    print("DFS: ", path_dfs)
    print("A*: ", path_a_star)

    draw_graph(graph, city_coordinates, path_bfs)
    draw_graph(graph, city_coordinates, path_dfs)
    draw_graph(graph, city_coordinates, path_a_star)
