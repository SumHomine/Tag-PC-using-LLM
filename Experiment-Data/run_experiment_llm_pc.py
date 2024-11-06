# from experiment_lmm_pc_andes import experiment_llm_pc     # comment in for andes data -> comment import above then out
from experiment_llm_pc import experiment_llm_pc
import os

#For Tu Server
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

# for data use data from .bif file in additionaldata
dataname = "win95pts" 
experiment_llm_pc(dataname, no_sid=True) #make True if Model is too complex for SID
