###code adapted from the github page: https://github.com/jliphard/DeepEvolve/blob/master/genome.py

"""The genome to be evolved."""

import random
import logging
import hashlib
import copy

from train import train_and_score

class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__(self, all_possible_genes = None, geneparam = {}, u_ID = 0, mom_ID = 0, dad_ID = 0, gen = 0):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Hyperparameters for the network, includes:
                gene_dense (list): [256, 512], 
                gene_conv_dropout (list): [0.25, 0.5],
                gene_dropout (list): [0.25, 0.5],
                gene_activation (list): ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'linear'],
                gene_optimizer (list):  ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
        """
        self.accuracy         = 0.0
        self.all_possible_genes = all_possible_genes
        self.geneparam        = geneparam #(dict): represents actual genome parameters
        self.u_ID             = u_ID
        self.parents          = [mom_ID, dad_ID]
        self.generation       = gen
        
        #hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()
        
    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """
        genh = str(self.geneparam['dense']) + str(self.geneparam['conv_dropout']) + str(self.geneparam['dropout']) + 
        self.geneparam['activation']  + self.geneparam['optimizer']

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

        self.accuracy = 0.0
            
    def set_genes_random(self):
        """Create a random genome.
        """
        self.parents = [0,0] 

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])
                
        self.update_hash()
        
    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            genome (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        #choose random gene to mutate
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        #perform mutation
        current_value    = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])
        
        possible_choices.remove(current_value)
        
        self.geneparam[gene_to_mutate] = random.choice(possible_choices)

        self.update_hash()
    
    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased
        """   
        self.generation = generation

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        """
        self.parents  = [mom_ID, dad_ID]
        
        self.geneparam = geneparam

        self.update_hash()

    def train(self, trainingset):
        """Train the genome and record the accuracy.

        Args:
            trainingset (str): Name of dataset to use.
        """
        if self.accuracy == 0.0: #don't bother retraining ones we already trained 
            self.accuracy = train_and_score(self, trainingset)

    def print_genome(self):
        """Print out a genome.
        """
        self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome.
        """
        self.print_geneparam()
        
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (self.accuracy * 100, self.u_ID, self.parents[0], 
                                                                           self.parents[1], self.generation))
        logging.info("Hash: %s" % self.hash)

    def print_geneparam(self):
        g = self.geneparam.copy()

        logging.info(g)