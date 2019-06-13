###code adapted from the github page: https://github.com/jliphard/DeepEvolve/blob/master/evolver.py

"""
Class that holds a genetic algorithm for evolving a genome.

Inspiration:

    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from __future__ import print_function

import random
import logging
import copy

from functools  import reduce
from operator   import add
from genome     import Genome
from idgen      import IDgen
from allgenomes import AllGenomes

class Evolver():
    """Class that implements genetic algorithm."""

    def __init__(self, all_possible_genes, retain=0.15, random_select=0.1, mutate_chance=0.3):
        """Create an optimizer.

        Args:
            all_possible_genes (dict): Possible genes (i.e., network
                hyperparameters)
            retain (float): Proportion of top-performing genomes to
                retain for the next generation
            random_select (float): Proportion of non-top-performing 
                genomes to retain for the next generation
            mutate_chance (float): Probability a genome will be
                randomly mutated

        """

        self.all_possible_genes = all_possible_genes
        self.retain             = retain
        self.random_select      = random_select
        self.mutate_chance      = mutate_chance

        #set the ID gen
        self.ids = IDgen()
        
    def create_population(self, count):
        """Create a population of random genomes.

        Args:
            count (int): Number of genomes to generate, aka the
                size of the population

        Returns:
            (list): Population of genome objects

        """
        pop = []

        i = 0

        while i < count:
            
            #initialize a new genome
            genome = Genome(self.all_possible_genes, {}, self.ids.get_next_ID(), 0, 0, self.ids.get_Gen())

            #set it to random parameters
            genome.set_genes_random()

            if i == 0:
                #this is where we will store all genomes
                self.master = AllGenomes(genome)
            else:
                #make sure it is unique....
                while self.master.is_duplicate(genome):
                    genome.mutate_one_gene()

            #add the genome to our population
            pop.append(genome)

            #and add to the master list
            if i > 0:
                self.master.add_genome(genome)

            i += 1

        return pop

    @staticmethod
    def fitness(genome):
        """Return the accuracy, which is our fitness function."""
        return genome.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of genomes

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(genome) for genome in pop))
        
        return summed / float((len(pop)))

    def breed(self, mom, dad):
        """Make two children from parental genes.

        Args:
            mother (dict): genome parameters
            father (dict): genome parameters

        Returns:
            (list): Two genome objects

        """
        children = []
        
        nr_genes = len(self.all_possible_genes)
        
        #get a gene location for single-point crossover
        crossover_loc = random.randint(1, nr_genes - 1) 

        child1 = {}
        child2 = {}

        #enforce defined genome order using list 
        keys = list(self.all_possible_genes)
        keys = sorted(keys) 

        #perform single-point crossover
        for x in range(0, nr_genes):
            if x < crossover_loc:
                child1[keys[x]] = mom.geneparam[keys[x]]
                child2[keys[x]] = dad.geneparam[keys[x]]
            else:
                child1[keys[x]] = dad.geneparam[keys[x]]
                child2[keys[x]] = mom.geneparam[keys[x]]

        #initialize a new genome
        #set its parameters to those just determined
        #they both have the same mom and dad
        genome1 = Genome( self.all_possible_genes, child1, self.ids.get_next_ID(), mom.u_ID, dad.u_ID, self.ids.get_Gen() )
        genome2 = Genome( self.all_possible_genes, child2, self.ids.get_next_ID(), mom.u_ID, dad.u_ID, self.ids.get_Gen() )

        #randomly mutate one gene
        if self.mutate_chance > random.random(): 
            genome1.mutate_one_gene()

        if self.mutate_chance > random.random(): 
            genome2.mutate_one_gene()

        #if child is a duplicate within the new generation, mutate a gene
        while self.master.is_duplicate(genome1):
            genome1.mutate_one_gene()

        self.master.add_genome(genome1)
        
        while self.master.is_duplicate(genome2):
            genome2.mutate_one_gene()

        self.master.add_genome(genome2)
        
        children.append(genome1)
        children.append(genome2)

        return children

    def evolve(self, pop):
        """Evolve a population of genomes.

        Args:
            pop (list): A list of genome parameters

        Returns:
            (list): The evolved population of genomes

        """
        #increase generation 
        self.ids.increase_Gen()

        #get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in pop]

        #and use those scores to fill in the master list
        for genome in pop:
            self.master.set_accuracy(genome)

        #sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        #get the number we want to keep unchanged for the next cycle
        retain_length = int(len(graded)*self.retain)

        #in this first step, we keep the 'top' X percent (as defined in self.retain)
        #we will not change them, except we will update the generation
        new_generation = graded[:retain_length]

        #for the lower scoring ones, randomly keep some anyway
        for genome in graded[retain_length:]:
            if self.random_select > random.random():
                randomgenome = copy.deepcopy(genome)
                
                while self.master.is_duplicate(randomgenome):
                    randomgenome.mutate_one_gene()

                randomgenome.set_generation(self.ids.get_Gen())
                new_generation.append(randomgenome)
                self.master.add_genome(randomgenome)
        
        #now find out how many spots we have left to fill in the new generation
        ng_length      = len(new_generation)

        desired_length = len(pop) - ng_length

        children       = []

        #add children, which are bred from pairs of genomes in the new generation (i.e. retained and randomly selected genomes)
        while len(children) < desired_length:

            #get a random mom and dad, but, need to make sure they are distinct
            parents  = random.sample(range(ng_length-1), k=2)
            
            i_male   = parents[0]
            i_female = parents[1]

            male   = new_generation[i_male]
            female = new_generation[i_female]

            #do crossover and mutation
            babies = self.breed(male, female)
            #the babies are guaranteed to be novel

            #add the children one at a time
            for baby in babies:
                children.append(baby)

        new_generation.extend(children)

        return new_generation