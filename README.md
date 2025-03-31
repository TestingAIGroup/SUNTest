# SUNTest
The Replication Package of SUNTest
This repository provides the replication package of SUNTest.

This manuscript introduces SUNTest, a novel test input generation approach designed to detect diverse faults and enhance the robustness of DNN models. SUNTest focuses on erroneous decision-making by localizing suspicious neurons responsible for misbehaviors through the execution spectrum analysis of neurons. To guide input mutations toward inducing diverse faults, SUNTest designs a hybrid fitness function that incorporates two types of feedback derived from neuron behaviors, including the fault-revealing capability of test inputs guided by suspicious neurons and the diversity of test inputs. Additionally, SUNTest adopts an adaptive selection strategy for mutation operators to prioritize operators likely to induce new fault types and improve the fitness value in each iteration.

# Details
SUNTest is implemented on an Ubuntu 18.04 server, using Python (v3.8) as the programming language. The open-source Machine Learning frameworks employed in this study are Keras (v2.5.0rc0) and TensorFlow (v2.5.0). The geatpy library (v2.6.0) is utilized to implement the search-based test input selection algorithm.

## Packages
SUNTest_app: Implements SUNTest’s test input generation algorithm. Specifically, test_generation.py represents the core algorithm of SUNTest, while test_generation_fitnessA.py, test_generation_fitnessB.py, test_generation_random_neurons.py, and test_generation_random_operators.py constitute its variant approaches.

coverage_analysis: Conducts coverage analysis of input sets, incorporating Distance-based Surprise Coverage (DSC) and NeuraL Coverage (NLC).

craft_dataset: Implements dataset processing techniques, including adversarial example generation and methodologies applied in research questions RQ1 to RQ5.

localization: Implements SUNTest's suspicious neuron localization algorithm. 

model_retraining: Contains code for retraining deep neural network (DNN) models.

MOON: https://github.com/TestingAIGroup/MOON. The online repository of our previous work, namely MOON. It focuses on selecting a subset of test inputs from an unlabeled test set for model retraining. MOON transforms test input selection into a search problem, employing a multi-objective optimization algorithm to identify test inputs
that maximize two objectives: suspicious neuron activation and input diversity.

