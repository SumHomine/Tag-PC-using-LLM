import numpy as np
import bnlearn as bn
import pandas as pd
import os
from tag_pc import tag_pc_from_true_skeleton
from tag_pc_utils import get_typelist_from_text, get_taglist_of_int_from_text
from visualization import create_graph_viz
from run_llm_tag_engineering import run_llm, run_llm_generic_prompt

#For Tu Server
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

types_forest = """
A : Producer
R : 1st Consumer
S : 1st Consumer
H : 2nd Consumer
B : 2nd Consumer
W : 3rd Consumer
F : 3rd Consumer
"""

type_insurance_default_0_topo = """
SocioEcon : First
GoodStudent : First
Age : First
RiskAversion : First
VehicleYear : Second
Accident : Fourth
ThisCarDam : Fourth
RuggedAuto : Third
MakeModel : Second
Antilock : Third
Mileage : First
DrivQuality : Fourth
DrivingSkill : Third
SeniorTrain : Second
ThisCarCost : Fifth
CarValue : Third
Theft : Fourth
AntiTheft : Second
HomeBase : Sixth
OtherCarCost : Fourth
PropCost : Fifth
OtherCar : First
MedCost : Fourth
Cushioning : Fourth
Airbag : Third
ILiCost : Fourth
DrivHist : Fourth
"""

# one vs all personal attribut
type_insurance_1vsall_1_person = """
SocioEcon : person
GoodStudent : person
Age : person
RiskAversion : person
VehicleYear : notperson
Accident : notperson
ThisCarDam : notperson
RuggedAuto : notperson
MakeModel : notperson
Antilock : notperson
Mileage : notperson
DrivQuality : person
DrivingSkill : person
SeniorTrain : person
ThisCarCost : notperson
CarValue : notperson
Theft : notperson
AntiTheft : notperson
HomeBase : notperson
OtherCarCost : notperson
PropCost : notperson
OtherCar : notperson
MedCost : notperson
Cushioning : notperson
Airbag : notperson
ILiCost : notperson
DrivHist : person
"""

# one vs all this car attribut
type_insurance_1vsall_2_car = """
SocioEcon : notcar
GoodStudent : notcar
Age : notcar
RiskAversion : notcar
VehicleYear : car
Accident : notcar
ThisCarDam : notcar
RuggedAuto : car
MakeModel : car
Antilock : car
Mileage : car
DrivQuality : notcar
DrivingSkill : notcar
SeniorTrain : notcar
ThisCarCost : car
CarValue : car
Theft : notcar
AntiTheft : car
HomeBase : notcar
OtherCarCost : notcar
PropCost : notcar
OtherCar : notcar
MedCost : notcar
Cushioning : car
Airbag : car
ILiCost : notcar
DrivHist : notcar
"""

# 1 vs all types: personal attributes, car attributes
tag_insurance_1vsall_0_person_car = """
SocioEcon : person, notcar
GoodStudent : person, notcar
Age : person, notcar
RiskAversion : person, notcar
VehicleYear : notperson, car
Accident : notperson, notcar
ThisCarDam : notperson, notcar
RuggedAuto : notperson, car
MakeModel : notperson, car
Antilock : notperson, car
Mileage : notperson, car
DrivQuality : person, notcar
DrivingSkill : person, notcar
SeniorTrain : person, notcar
ThisCarCost : notperson, car
CarValue : notperson, car
Theft : notperson, notcar
AntiTheft : notperson, car
HomeBase : notperson, notcar
OtherCarCost : notperson, notcar
PropCost : notperson, notcar
OtherCar : notperson, notcar
MedCost : notperson, notcar
Cushioning : notperson, car
Airbag : notperson, car
ILiCost : notperson, notcar
DrivHist : person, notcar
"""

# topological typing + 1 vs all types: personal attributes, car attributes
tag_insurance_1_topo_person_car = """
SocioEcon : first, person, notcar
GoodStudent : first, person, notcar
Age : first, person, notcar
RiskAversion : first, person, notcar
VehicleYear : second, notperson, car
Accident : fourth, notperson, notcar
ThisCarDam : fourth, notperson, notcar
RuggedAuto : third, notperson, car
MakeModel : second, notperson, car
Antilock : third, notperson, car
Mileage : first, notperson, car
DrivQuality : fourth, person, notcar
DrivingSkill : third, person, notcar
SeniorTrain : second, person, notcar
ThisCarCost : fifth, notperson, car
CarValue : third, notperson, car
Theft : fourth, notperson, notcar
AntiTheft : second, notperson, car
HomeBase : sixth, notperson, notcar
OtherCarCost : fourth, notperson, notcar
PropCost : fifth, notperson, notcar
OtherCar : first, notperson, notcar
MedCost : fourth, notperson, notcar
Cushioning : fourth, notperson, car
Airbag : third, notperson, car
ILiCost : fourth, notperson, notcar
DrivHist : fourth, person, notcar
"""

tag_insurance_ai_generated_1 = """
SocioEcon : Demographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
GoodStudent : Demographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
Age : Demographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
RiskAversion : notDemographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, Risk Factors
VehicleYear : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
Accident : notDemographics, notVehicle Characteristics, Accident Characteristics, notFinancial Factors, notRisk Factors
ThisCarDam : notDemographics, notVehicle Characteristics, Accident Characteristics, notFinancial Factors, notRisk Factors
RuggedAuto : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
MakeModel : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
Antilock : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
Mileage : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
DrivQuality : notDemographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, Risk Factors
DrivingSkill : notDemographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, Risk Factors
SeniorTrain : Demographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
ThisCarCost : notDemographics, notVehicle Characteristics, notAccident Characteristics, Financial Factors, notRisk Factors
CarValue : notDemographics, notVehicle Characteristics, notAccident Characteristics, Financial Factors, notRisk Factors
Theft : notDemographics, notVehicle Characteristics, Accident Characteristics, notFinancial Factors, notRisk Factors
AntiTheft : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
HomeBase : Demographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
OtherCarCost : notDemographics, notVehicle Characteristics, notAccident Characteristics, Financial Factors, notRisk Factors
PropCost : notDemographics, notVehicle Characteristics, notAccident Characteristics, Financial Factors, notRisk Factors
OtherCar : notDemographics, Vehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
MedCost : notDemographics, notVehicle Characteristics, notAccident Characteristics, Financial Factors, notRisk Factors
Cushioning : notDemographics, notVehicle Characteristics, Accident Characteristics, notFinancial Factors, notRisk Factors
Airbag : notDemographics, notVehicle Characteristics, Accident Characteristics, notFinancial Factors, notRisk Factors
ILiCost : notDemographics, notVehicle Characteristics, notAccident Characteristics, Financial Factors, notRisk Factors
DrivHist : Demographics, notVehicle Characteristics, notAccident Characteristics, notFinancial Factors, notRisk Factors
"""

tag_insurance_ai_generated_2 = """
SocioEcon : Demographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
GoodStudent : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, Insurance-related, notOther
Age : Demographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
RiskAversion : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, Insurance-related, notOther
VehicleYear : notDemographics, Vehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
Accident : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
ThisCarDam : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, Accident-related, notInsurance-related, notOther
RuggedAuto : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, Accident-related, notInsurance-related, notOther
MakeModel : notDemographics, Vehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
Antilock : notDemographics, Vehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
Mileage : notDemographics, Vehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
DrivQuality : notDemographics, notVehicle Characteristics, Driving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
DrivingSkill : notDemographics, notVehicle Characteristics, Driving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
SeniorTrain : Demographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
ThisCarCost : notDemographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
CarValue : notDemographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
Theft : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, Accident-related, notInsurance-related, notOther
AntiTheft : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, Accident-related, notInsurance-related, notOther
HomeBase : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, Other
OtherCarCost : notDemographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
PropCost : notDemographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
OtherCar : notDemographics, notVehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, Other
MedCost : notDemographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
Cushioning : notDemographics, Vehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
Airbag : notDemographics, Vehicle Characteristics, notDriving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
ILiCost : notDemographics, notVehicle Characteristics, notDriving Habits, Financial, notAccident-related, notInsurance-related, notOther
DrivHist : notDemographics, notVehicle Characteristics, Driving Habits, notFinancial, notAccident-related, notInsurance-related, notOther
"""

tag_insurance_ai_generated_cycle = """
SocioEcon : notVehicle-related, notDriver-related, notAccident-related, Financial, Demographic, notSafety-related
GoodStudent : notVehicle-related, notDriver-related, notAccident-related, notFinancial, Demographic, notSafety-related
Age : notVehicle-related, notDriver-related, notAccident-related, notFinancial, Demographic, notSafety-related
RiskAversion : notVehicle-related, Driver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
VehicleYear : Vehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
Accident : notVehicle-related, notDriver-related, Accident-related, notFinancial, notDemographic, notSafety-related
ThisCarDam : Vehicle-related, notDriver-related, Accident-related, notFinancial, notDemographic, notSafety-related
RuggedAuto : Vehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
MakeModel : Vehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
Antilock : Vehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, Safety-related
Mileage : Vehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
DrivQuality : notVehicle-related, Driver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
DrivingSkill : notVehicle-related, Driver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
SeniorTrain : notVehicle-related, notDriver-related, notAccident-related, notFinancial, Demographic, notSafety-related
ThisCarCost : Vehicle-related, notDriver-related, notAccident-related, Financial, notDemographic, notSafety-related
CarValue : Vehicle-related, notDriver-related, notAccident-related, Financial, notDemographic, notSafety-related
Theft : notVehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, Safety-related
AntiTheft : notVehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, Safety-related
HomeBase : notVehicle-related, notDriver-related, notAccident-related, notFinancial, Demographic, notSafety-related
OtherCarCost : notVehicle-related, notDriver-related, notAccident-related, Financial, notDemographic, notSafety-related
PropCost : notVehicle-related, notDriver-related, notAccident-related, Financial, notDemographic, notSafety-related
OtherCar : Vehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
MedCost : notVehicle-related, notDriver-related, notAccident-related, Financial, notDemographic, notSafety-related
Cushioning : notVehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, Safety-related
Airbag : notVehicle-related, notDriver-related, notAccident-related, notFinancial, notDemographic, Safety-related
ILiCost : notVehicle-related, notDriver-related, notAccident-related, Financial, notDemographic, notSafety-related
DrivHist : notVehicle-related, Driver-related, notAccident-related, notFinancial, notDemographic, notSafety-related
"""

tag_insurance_ai_generated_domain_old = """
SocioEcon : Demographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
GoodStudent : Demographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
Age : Demographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
RiskAversion : Demographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
VehicleYear : notDemographics, Vehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
Accident : notDemographics, notVehicle Characteristics, Accident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
ThisCarDam : notDemographics, Vehicle Characteristics, Accident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
RuggedAuto : notDemographics, Vehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
MakeModel : notDemographics, Vehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
Antilock : notDemographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, Safety Features, notInsurance-Related
Mileage : notDemographics, Vehicle Characteristics, notAccident Details, notFinancial Situation, Driving Habits, notSafety Features, notInsurance-Related
DrivQuality : notDemographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, Driving Habits, notSafety Features, notInsurance-Related
DrivingSkill : notDemographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, Driving Habits, notSafety Features, notInsurance-Related
SeniorTrain : Demographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
ThisCarCost : notDemographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
CarValue : notDemographics, Vehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
Theft : notDemographics, notVehicle Characteristics, Accident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
AntiTheft : notDemographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, Safety Features, notInsurance-Related
HomeBase : Demographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
OtherCarCost : notDemographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
PropCost : notDemographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
OtherCar : notDemographics, Vehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
MedCost : notDemographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
Cushioning : notDemographics, notVehicle Characteristics, Accident Details, notFinancial Situation, notDriving Habits, notSafety Features, notInsurance-Related
Airbag : notDemographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, notDriving Habits, Safety Features, notInsurance-Related
ILiCost : notDemographics, notVehicle Characteristics, notAccident Details, Financial Situation, notDriving Habits, notSafety Features, notInsurance-Related
DrivHist : notDemographics, notVehicle Characteristics, notAccident Details, notFinancial Situation, Driving Habits, notSafety Features, notInsurance-Related
"""

tag_insurance_ai_generated_domain = """
SocioEcon : Demographic, notVehicle, notRisk, notFinancial, notSafety
GoodStudent : notDemographic, notVehicle, Risk, notFinancial, notSafety
Age : Demographic, notVehicle, notRisk, notFinancial, notSafety
RiskAversion : notDemographic, notVehicle, Risk, notFinancial, notSafety
VehicleYear : notDemographic, Vehicle, notRisk, notFinancial, notSafety
Accident : notDemographic, notVehicle, Risk, notFinancial, notSafety
ThisCarDam : notDemographic, notVehicle, notRisk, notFinancial, Safety
RuggedAuto : notDemographic, Vehicle, notRisk, notFinancial, notSafety
MakeModel : notDemographic, Vehicle, notRisk, notFinancial, notSafety
Antilock : notDemographic, Vehicle, notRisk, notFinancial, notSafety
Mileage : notDemographic, Vehicle, notRisk, notFinancial, notSafety
DrivQuality : notDemographic, notVehicle, Risk, notFinancial, notSafety
DrivingSkill : notDemographic, notVehicle, Risk, notFinancial, notSafety
SeniorTrain : Demographic, notVehicle, notRisk, notFinancial, notSafety
ThisCarCost : notDemographic, notVehicle, notRisk, Financial, notSafety
CarValue : notDemographic, notVehicle, notRisk, Financial, notSafety
Theft : notDemographic, notVehicle, notRisk, notFinancial, Safety
AntiTheft : notDemographic, notVehicle, notRisk, notFinancial, Safety
HomeBase : Demographic, notVehicle, notRisk, notFinancial, notSafety
OtherCarCost : notDemographic, notVehicle, notRisk, Financial, notSafety
PropCost : notDemographic, notVehicle, notRisk, Financial, notSafety
OtherCar : notDemographic, Vehicle, notRisk, notFinancial, notSafety
MedCost : notDemographic, notVehicle, notRisk, Financial, notSafety
Cushioning : notDemographic, notVehicle, notRisk, notFinancial, Safety
Airbag : notDemographic, Vehicle, notRisk, notFinancial, notSafety
ILiCost : notDemographic, notVehicle, notRisk, Financial, notSafety
DrivHist : notDemographic, notVehicle, Risk, notFinancial, notSafety
"""

tag_insurance_ai_generated_generic = """
SocioEcon : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
GoodStudent : notCausal, Demographic, notEconomic, notVehicle, notSafety, Behavioral, notEnvironmental
Age : notCausal, Demographic, notEconomic, notVehicle, notSafety, notBehavioral, notEnvironmental
RiskAversion : notCausal, notDemographic, notEconomic, notVehicle, notSafety, Behavioral, notEnvironmental
VehicleYear : notCausal, notDemographic, notEconomic, Vehicle, notSafety, notBehavioral, notEnvironmental
Accident : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
ThisCarDam : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
RuggedAuto : notCausal, notDemographic, notEconomic, Vehicle, notSafety, notBehavioral, notEnvironmental
MakeModel : notCausal, notDemographic, notEconomic, Vehicle, notSafety, notBehavioral, notEnvironmental
Antilock : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
Mileage : notCausal, notDemographic, notEconomic, Vehicle, notSafety, notBehavioral, notEnvironmental
DrivQuality : notCausal, notDemographic, notEconomic, notVehicle, notSafety, Behavioral, notEnvironmental
DrivingSkill : notCausal, notDemographic, notEconomic, notVehicle, notSafety, Behavioral, notEnvironmental
SeniorTrain : notCausal, Demographic, notEconomic, notVehicle, notSafety, Behavioral, notEnvironmental
ThisCarCost : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
CarValue : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
Theft : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
AntiTheft : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
HomeBase : notCausal, notDemographic, notEconomic, notVehicle, notSafety, notBehavioral, Environmental
OtherCarCost : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
PropCost : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
OtherCar : notCausal, notDemographic, notEconomic, Vehicle, notSafety, notBehavioral, notEnvironmental
MedCost : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
Cushioning : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
Airbag : notCausal, notDemographic, notEconomic, notVehicle, Safety, notBehavioral, notEnvironmental
ILiCost : notCausal, notDemographic, Economic, notVehicle, notSafety, notBehavioral, notEnvironmental
DrivHist : notCausal, notDemographic, notEconomic, notVehicle, notSafety, Behavioral, notEnvironmental
"""

asia_type_topological = """
    asia : first
    tub : first
    smoke : second
    lung : second
    bronc : third
    either : third
    xray : fourth
    dysp : fourth
    """

asia_type_risk = """
    asia : risk
    tub : notrisk
    smoke : risk
    lung : notrisk
    bronc : notrisk
    either : notrisk
    xray : notrisk
    dysp : notrisk
"""

#not type consistent
asia_type2 = """ 
    asia : nosick
    tub : sick
    smoke : nosick
    lung : sick
    bronc : sick
    either : sick
    xray : nosick
    dysp : sick
    """

asia_tag_intuitive = """
    asia : notsick, risk
    tub : sick, notrisk
    smoke : notsick, risk
    lung : sick, notrisk
    bronc : sick, notrisk
    either : sick, notrisk
    xray : notsick, notrisk
    dysp : sick, notrisk
"""


asia_tag1 = """
    asia : first, nosick
    tub : first, sick
    smoke : second, nosick
    lung : second, sick
    bronc : third, sick
    either : third, sick
    xray : fourth, nosick
    dysp : fourth, sick
    """

asia_tag1_2then1 = """
    asia : nosick, first
    tub : sick, first
    smoke : nosick, second
    lung : sick, second
    bronc : sick, third
    either : sick, third
    xray : nosick, fourth
    dysp : sick, fourth
    """

asia_tag2_topological_onevsall = """
    asia : first, notsecond, nothird, notfourth, nosick
    tub : first, notsecond, nothird, notfourth, sick
    smoke : notfirst, second, nothird, notfourth, nosick
    lung : notfirst, second, nothird, notfourth, sick
    bronc : notfirst, notsecond, thrid, notfourth, sick
    either : notfirst, notsecond, third, notfourth, sick
    xray : notfirst, notsecond, nothird, fourth, nosick
    dysp : notfirst, notsecond, nothird, fourth, sick
    """

asia_tag_ai_generated = """
    asia : Health-related, Risk factor, notSymptom, notDiagnostic tool, notEnvironmental
    tub : Health-related, notRisk factor, Symptom, notDiagnostic tool, notEnvironmental
    smoke : notHealth-related, Risk factor, notSymptom, notDiagnostic tool, Environmental
    lung : Health-related, notRisk factor, Symptom, notDiagnostic tool, notEnvironmental
    bronc : Health-related, notRisk factor, Symptom, notDiagnostic tool, notEnvironmental
    either : notHealth-related, notRisk factor, notSymptom, notDiagnostic tool, notEnvironmental
    xray : notHealth-related, notRisk factor, notSymptom, Diagnostic tool, notEnvironmental
    dysp : notHealth-related, notRisk factor, Symptom, notDiagnostic tool, notEnvironmental
    """

asia_tag_ai_experiment_generic = """
    asia : Health-related, Respiratory, notDiagnostic, notRisk factor
    tub : Health-related, Respiratory, Diagnostic, notRisk factor
    smoke : Health-related, notRespiratory, notDiagnostic, Risk factor
    lung : Health-related, Respiratory, notDiagnostic, notRisk factor
    bronc : Health-related, Respiratory, notDiagnostic, notRisk factor
    either : notHealth-related, notRespiratory, Diagnostic, notRisk factor
    xray : notHealth-related, notRespiratory, Diagnostic, notRisk factor
    dysp : Health-related, Respiratory, notDiagnostic, notRisk factor
    """

asia_tag_ai_experiment_domain = """
    asia : Infectious, notCancer, notRespiratory, RiskFactor, notDiagnostic
    tub : Infectious, notCancer, Respiratory, notRiskFactor, notDiagnostic
    smoke : notInfectious, notCancer, Respiratory, RiskFactor, notDiagnostic
    lung : notInfectious, Cancer, Respiratory, notRiskFactor, notDiagnostic
    bronc : notInfectious, notCancer, Respiratory, notRiskFactor, notDiagnostic
    either : notInfectious, notCancer, notRespiratory, notRiskFactor, Diagnostic
    xray : notInfectious, notCancer, notRespiratory, notRiskFactor, Diagnostic
    dysp : notInfectious, notCancer, Respiratory, notRiskFactor, notDiagnostic
    """


tag_sprinkler = """
    Cloudy : Weather, Weather, NotWatering
    Sprinkler : Watering, NotWeather, Watering
    Rain : Weather, Weather, Watering
    Wet_Grass : Plant_Con, NotWeather, NotWatering   
    """


tag_sprinkler_bad_majority = """
    Cloudy : NotWatering, Watervapor, Weather
    Sprinkler : Watering, Watervapor, NotWeather
    Rain : Watering, NotWatervapor, Weather
    Wet_Grass : NotWatering, NotWatervapor, NotWeather  
    """


tag_sprinkler_weather_watervapor = """
    Cloudy : Weather, Uses Watervapor
    Sprinkler : NotWeather, Uses Watervapor
    Rain : Weather, NotUses Watervapor
    Wet_Grass : NotWeather, NotUses Watervapor
    """

tag_sprinkler_generic_ai = """
    Cloudy : Weather, notMan-made, Natural, notState
    Sprinkler : Weather, Man-made, notNatural, notState
    Rain : Weather, notMan-made, Natural, notState
    Wet_Grass : notWeather, notMan-made, Natural, State
"""

# node names for generic prompt LLM Tagging
node_names_sprinkler = ["Cloudy", "Sprinkler", "Rain", "Wet_Grass"]
node_names_asia = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray",  "dysp"]
node_names_alarm = ['LVFAILURE', 'HISTORY', 'LVEDVOLUME', 'CVP', 'PCWP', 'HYPOVOLEMIA', 'STROKEVOLUME', 'ERRLOWOUTPUT', 'HRBP', 'HR', 'ERRCAUTER', 'HREKG', 'HRSAT', 'ANAPHYLAXIS', 'TPR', 'ARTCO2', 'EXPCO2', 'VENTLUNG', 'INTUBATION', 'MINVOL', 'FIO2', 'PVSAT', 'VENTALV', 'SAO2', 'SHUNT', 'PULMEMBOLUS', 'PAP', 'PRESS', 'KINKEDTUBE', 'VENTTUBE', 'MINVOLSET', 'VENTMACH', 'DISCONNECT', 'CATECHOL', 'INSUFFANESTH', 'CO', 'BP']
node_names_barley = ['komm', 'nedbarea', 'jordtype', 'nmin', 'aar_mod', 'potnmin', 'forfrugt', 'jordn', 'exptgens', 'pesticid', 'mod_nmin', 'ngodnt', 'nopt', 'ngodnn', 'ngodn', 'nprot', 'rokap', 'saatid', 'dgv1059', 'sort', 'srtprot', 'dg25', 'ngtilg', 'nplac', 'ntilg', 'saamng', 'saakern', 'tkvs', 'frspdag', 'jordinf', 'partigerm', 'markgrm', 'antplnt', 'sorttkv', 'aks_m2', 'keraks', 'dgv5980', 'aks_vgt', 'srtsize', 'ksort', 'protein', 'udb', 'spndx', 'tkv', 'slt22', 's2225', 's2528', 'bgbyg']
node_names_hepar = ['hepatotoxic', 'THepatitis', 'alcoholism', 'gallstones', 'choledocholithotomy', 'hospital', 'injections', 'surgery', 'transfusion', 'ChHepatitis', 'vh_amn', 'sex', 'PBC', 'age', 'fibrosis', 'diabetes', 'obesity', 'Steatosis', 'Cirrhosis', 'Hyperbilirubinemia', 'triglycerides', 'RHepatitis', 'fatigue', 'bilirubin', 'itching', 'upper_pain', 'fat', 'pain_ruq', 'pressure_ruq', 'phosphatase', 'skin', 'ama', 'le_cells', 'joints', 'pain', 'proteins', 'edema', 'platelet', 'inr', 'bleeding', 'flatulence', 'alcohol', 'encephalopathy', 'urea', 'ascites', 'hepatomegaly', 'hepatalgia', 'density', 'ESR', 'alt', 'ast', 'amylase', 'ggtp', 'cholesterol', 'hbsag', 'hbsag_anti', 'anorexia', 'nausea', 'spleen', 'consciousness', 'spiders', 'jaundice', 'albumin', 'edge', 'irregular_liver', 'hbc_anti', 'hcv_anti', 'palms', 'hbeag', 'carcinoma']
node_names_sachs = ['Erk', 'Akt', 'PKA', 'Mek', 'Jnk', 'PKC', 'Raf', 'P38', 'PIP3', 'PIP2', 'Plcg']
node_names_insurance = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']
node_names = ['DISPLACEM0', 'RApp1', 'SNode_3', 'GIVEN_1', 'RApp2', 'SNode_8', 'SNode_16', 'SNode_20', 'NEED1', 'SNode_21', 'GRAV2', 'SNode_24', 'VALUE3', 'SNode_15', 'SNode_25', 'SLIDING4', 'SNode_11', 'SNode_26', 'CONSTANT5', 'SNode_47', 'VELOCITY7', 'KNOWN6', 'RApp3', 'KNOWN8', 'RApp4', 'SNode_27', 'GOAL_2', 'GOAL_48', 'COMPO16', 'TRY12', 'TRY11', 'SNode_5', 'GOAL_49', 'SNode_6', 'GOAL_50', 'CHOOSE19', 'SNode_17', 'SNode_51', 'SYSTEM18', 'SNode_52', 'KINEMATI17', 'GOAL_53', 'IDENTIFY10', 'SNode_28', 'IDENTIFY9', 'TRY13', 'TRY14', 'TRY15', 'SNode_29', 'VAR20', 'SNode_31', 'SNode_10', 'SNode_33', 'GIVEN21', 'SNode_34', 'GOAL_56', 'APPLY32', 'GOAL_57', 'CHOOSE35', 'SNode_7', 'SNode_59', 'MAXIMIZE34', 'SNode_60', 'AXIS33', 'GOAL_61', 'WRITE31', 'GOAL_62', 'WRITE30', 'GOAL_63', 'RESOLVE37', 'SNode_64', 'NEED36', 'SNode_9', 'SNode_41', 'SNode_42', 'SNode_43', 'IDENTIFY39', 'GOAL_66', 'RESOLVE38', 'SNode_67', 'SNode_54', 'IDENTIFY41', 'GOAL_69', 'RESOLVE40', 'SNode_70', 'SNode_55', 'IDENTIFY43', 'GOAL_72', 'RESOLVE42', 'SNode_73', 'SNode_74', 'KINE29', 'SNode_4', 'SNode_75', 'VECTOR44', 'GOAL_79', 'EQUATION28', 'VECTOR27', 'RApp5', 'GOAL_80', 'RApp6', 'GOAL_81', 'TRY25', 'TRY24', 'GOAL_83', 'GOAL_84', 'CHOOSE47', 'SNode_86', 'SYSTEM46', 'SNode_156', 'NEWTONS45', 'GOAL_98', 'DEFINE23', 'SNode_37', 'IDENTIFY22', 'TRY26', 'SNode_38', 'SNode_40', 'SNode_44', 'SNode_46', 'SNode_65', 'NULL48', 'SNode_68', 'SNode_71', 'GOAL_87', 'FIND49', 'SNode_88', 'NORMAL50', 'NORMAL52', 'INCLINE51', 'SNode_91', 'SNode_12', 'SNode_13', 'STRAT_90', 'HORIZ53', 'BUGGY54', 'SNode_92', 'SNode_93', 'IDENTIFY55', 'SNode_94', 'WEIGHT56', 'SNode_95', 'WEIGHT57', 'SNode_97', 'GOAL_99', 'FIND58', 'SNode_100', 'IDENTIFY59', 'SNode_102', 'FORCE60', 'GOAL_103', 'APPLY61', 'GOAL_104', 'CHOOSE62', 'SNode_106', 'SNode_152', 'GOAL_107', 'WRITE63', 'GOAL_108', 'WRITE64', 'GOAL_109', 'GOAL_110', 'GOAL65', 'GOAL_111', 'GOAL66', 'NEED67', 'RApp7', 'RApp8', 'SNode_112', 'GOAL_113', 'GOAL68', 'GOAL_114', 'SNode_115', 'SNode_116', 'VECTOR69', 'SNode_117', 'SNode_118', 'VECTOR70', 'SNode_119', 'EQUAL71', 'SNode_120', 'GOAL_121', 'GOAL72', 'SNode_122', 'SNode_123', 'VECTOR73', 'SNode_124', 'NEWTONS74', 'SNode_125', 'SUM75', 'GOAL_126', 'GOAL_127', 'RApp9', 'RApp10', 'SNode_128', 'GOAL_129', 'GOAL_130', 'SNode_131', 'SNode_132', 'SNode_133', 'SNode_134', 'SNode_135', 'SNode_154', 'SNode_136', 'SNode_137', 'GOAL_142', 'GOAL_143', 'GOAL_146', 'RApp11', 'RApp12', 'RApp13', 'GOAL_147', 'GOAL_149', 'TRY76', 'GOAL_150', 'APPLY77', 'SNode_151', 'GRAV78', 'GOAL_153', 'SNode_155', 'SNode_14', 'SNode_18', 'SNode_19']


data = None
# for reading from csv (for for example forest data)
# Read CSV file and extract node names from first line 
df = pd.read_csv("Tag-PC-using-LLM/generated_forestdata.csv", header=None)
node_names = df.iloc[0].values
# Remove first row and get data as nd.array
forest_data = df.iloc[1:].values
forest_data = forest_data = df.iloc[1:].values.astype(int)  # Convert all columns to int for forest



# Change Here
# data = forest_data #comment back in when using forest
dataname = "asia" #see possible Strings in tpc_from_true_skeleton (TODO)
llm_generated_tags = False #when False, check assigned tag in line 288 #TODO check line number before publishing

if llm_generated_tags:
    # tags, node_names = run_llm(dataname, deterministic=True) # get tags via LLM for implemented cases
    tags, node_names = run_llm_generic_prompt(node_names=node_names_barley, determinstic=True) # comment out for getting LLM tags via generic prompt #XXX you need top update node_names depending on dataset
else:
    tags = asia_tag1 #XXX change tag depending on data, see tags above 

equal_majority_rule_tagged = False #true means majority tag, false is weighted tag
majority_rule_typed = True #majority rule of typing algo, true is normaly the better choice

dag, stat_tests, node_names, taglist = tag_pc_from_true_skeleton(dataname=dataname, tags=tags, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed, data=data)


dir = "Tag-PC-using-LLM/tagged-PC"
fname = "tdag_" + dataname + "_" + "true_skeleton_" + ("AI_Tag_New_" if llm_generated_tags else "") + ("majoritytag_" if equal_majority_rule_tagged else "weightedtag_") + ("majoritytype" if majority_rule_typed else "naivetype") + "_0" # + "_multitag1_topo_person_car" # _weather_watervapor
create_graph_viz(dag=dag, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname) #print using first tag #TODO tagging sichtbarer machen
if llm_generated_tags: print(f"Ai generated Tags:\n{tags}")  # for debuging TODO ggf. remove for publishen