# OTSU thresholding based on Genetic Algorithm
import cv2
import numpy as np

def otsu(image_path, output_path):
    # Read image
    img = cv2.imread(image_path, 0)
    # convert to grayscale image
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # use Genetic Algorithm to find the best threshold
    threshold = genetic_algorithm(gray)
    # threshold = 127

    # apply thresholding
    binary = thresholding(gray, threshold)

    # save the image
    cv2.imwrite(output_path, binary)

def genetic_algorithm(gray):
    # initialize the population
    population = np.random.randint(0, 255, size=(10, 1))
    iteration = 5

    # calculate the fitness of each individual
    fitness = fitness_function(gray, population)
    # selection
    population = selection(population, fitness)
    # crossover
    population = crossover(population)
    # mutation
    population = mutation(population)

    for i in range(iteration):
        fitness = fitness_function(gray, population)
        population = selection(population, fitness)
        population = crossover(population)
        population = mutation(population)

    # return the best threshold
    return population[np.argmax(fitness)]

def fitness_function(gray, population):
    # calculate the fitness of the individual
    fitness = []
    for threshold in population:
        # count the number of pixels of background and foreground
        background = gray[gray <= threshold]
        foreground = gray[gray > threshold]
        # calculate the probability of background and foreground
        p_background = len(background) / (gray.shape[0] * gray.shape[1])
        p_foreground = len(foreground) / (gray.shape[0] * gray.shape[1])
        # calculate the mean of background and foreground
        mean_background = np.mean(background)
        mean_foreground = np.mean(foreground)
        # calculate the variance as the fitness
        fitness.append(p_background * p_foreground * (mean_background - mean_foreground) ** 2)

    fitness = np.array(fitness)
    fitness = np.where(np.isnan(fitness), 0, fitness)

    return fitness

def selection(population, fitness):
    # wheel selection
    fitness = fitness / np.sum(fitness)
    fitness = np.cumsum(fitness)
    new_population = []
    for i in range(len(population)):
        r = np.random.rand()
        for j, value in enumerate(fitness):
            if r < value:
                new_population.append(population[j])
                break
    return np.array(new_population)

def crossover(population):
    # crossover
    new_population = []
    for i in range(len(population)):
        r = np.random.rand()
        # crossover rate
        rate = 0.8
        if r < rate:
            # select parents
            parent1 = population[i]
            parent2 = population[np.random.randint(0, len(population))]
            # choose the crossover point
            point = np.random.randint(0, 256)
            # crossover operation
            child1 = (parent1 & point) | (parent2 & (255 - point))
            child2 = (parent2 & point) | (parent1 & (255 - point))
            new_population.append(child1)
            new_population.append(child2)
        else:
            new_population.append(population[i])
    return np.array(new_population)
            
def mutation(population):
    # mutation
    new_population = []
    for i in range(len(population)):
        r = np.random.rand()
        # mutation rate
        rate = 0.05
        if r < rate:
            # choose the mutation point
            point = np.random.randint(0, 256)
            # mutation operation
            child = population[i] ^ point
            new_population.append(child)
        else:
            new_population.append(population[i])
    return np.array(new_population)

def thresholding(gray, threshold):
    # apply thresholding
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[gray > threshold] = 255

    return binary

if __name__ == '__main__':
    image_path = './ImageConvert/origin.jpg'
    output_path = './ImageConvert/otsu.jpg'
    otsu(image_path, output_path)
    # show the original image and the result in the same 600*300 window 
    img = cv2.imread(image_path)
    binary = cv2.imread(output_path)
    cv2.namedWindow('OTSU', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('OTSU', 600, 300)
    cv2.imshow('OTSU', np.hstack((img, binary)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

