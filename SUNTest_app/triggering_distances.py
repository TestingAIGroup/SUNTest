
import numpy as np


class TriggeringInfo:

    def __init__(self):
        # store the information of all triggering inputs
        self.trigger_input_archive = []
        self.trigger_inputs_ats_archive = []
        self.fault_type_archive = []
        self.average_time_archive = {}
        self.all_trigger_inps_archive = {}
        self.all_gen_inps = {}



