# SUNTest
The Replication Package of SUNTest
This repository provides the replication package of SUNTest.

This manuscript introduces SUNTest, a novel test input generation approach designed to detect diverse faults and enhance the robustness of DNN models. SUNTest focuses on erroneous decision-making by localizing suspicious neurons responsible for misbehaviors through the execution spectrum analysis of neurons. To guide input mutations toward inducing diverse faults, SUNTest designs a hybrid fitness function that incorporates two types of feedback derived from neuron behaviors, including the fault-revealing capability of test inputs guided by suspicious neurons and the diversity of test inputs. Additionally, SUNTest adopts an adaptive selection strategy for mutation operators to prioritize operators likely to induce new fault types and improve the fitness value in each iteration.

# Details

SUNTest_app: The implementation of SUNTest's test input generation algorithm.

coverage_analysis: Distance-based Surprise Coverage (DSC) and NeuraL Coverage (NLC) analysis of input sets.

craft_dataset: The implementation of dataset treatments, including adversarial example generation and treatments adopted in RQ1 to RQ5.

localization: The implementation of suspicious neuron localization.

model_retraining: The code for retraining DNN models.
