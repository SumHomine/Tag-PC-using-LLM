from llm_interface import load_model_pipeline, text_generation, text_reduction_taglist, text_reduction_variable_tags
from tag_pc_utils import get_taglist_from_llm_output, recover_tag_string_onevsall_using_taglist

def run_llm(data : str, deterministic = False):
    """
    use this method to get tags using a LLM 
    :param data: string of the dataset you want to use, this method will then automatically use the correct prompts (including node_names)
    :Param deterministic: if true try to make model as deterministic as possible, i.e. the same prompt returns the same generated text
    """ #TODO returns einfügen

    #match data and get node_names and initial prompt for the given data
    match data:
        case "forest":
            print("TODO IMPLEMENT") #TODO Implement
            node_names = ["A","R","S","H","B","W","F"]
            generation_prompt = """"""
            return
        case "asia":
            node_names = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray",  "dysp"]
            generation_prompt = f"""We have found a causal system consisting of {len(node_names)} variables. The model can be described as follows:
            Shortness-of-breath (dyspnoea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking is known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.
            Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
            # description see lauritzen et al. 1988
        case "barley":
            node_names = ['komm', 'nedbarea', 'jordtype', 'nmin', 'aar_mod', 'potnmin', 'forfrugt', 'jordn', 'exptgens', 'pesticid', 'mod_nmin', 'ngodnt', 'nopt', 'ngodnn', 'ngodn', 'nprot', 'rokap', 'saatid', 'dgv1059', 'sort', 'srtprot', 'dg25', 'ngtilg', 'nplac', 'ntilg', 'saamng', 'saakern', 'tkvs', 'frspdag', 'jordinf', 'partigerm', 'markgrm', 'antplnt', 'sorttkv', 'aks_m2', 'keraks', 'dgv5980', 'aks_vgt', 'srtsize', 'ksort', 'protein', 'udb', 'spndx', 'tkv', 'slt22', 's2225', 's2528', 'bgbyg']
            generation_prompt = f"""We have found a causal system consisting of {len(node_names)} variables. The model can be described as follows:
            A preliminary model for the production of beer from Danish malting barley grown without pesticides.
            Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
            # description see https://bnma.co/bn/80        
        case "insurance":
            node_names = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']
            generation_prompt = f"""We have found a causal system consisting of {len(node_names)} variables. The system can be described as follows:
            This system describes a risk estimation used by a car insurance company. So that each variable impacts the likelihood of a car accident.
            Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
            # description selfmade inspired by paper
        case "andes":
            node_names = ['DISPLACEM0', 'RApp1', 'SNode_3', 'GIVEN_1', 'RApp2', 'SNode_8', 'SNode_16', 'SNode_20', 'NEED1', 'SNode_21', 'GRAV2', 'SNode_24', 'VALUE3', 'SNode_15', 'SNode_25', 'SLIDING4', 'SNode_11', 'SNode_26', 'CONSTANT5', 'SNode_47', 'VELOCITY7', 'KNOWN6', 'RApp3', 'KNOWN8', 'RApp4', 'SNode_27', 'GOAL_2', 'GOAL_48', 'COMPO16', 'TRY12', 'TRY11', 'SNode_5', 'GOAL_49', 'SNode_6', 'GOAL_50', 'CHOOSE19', 'SNode_17', 'SNode_51', 'SYSTEM18', 'SNode_52', 'KINEMATI17', 'GOAL_53', 'IDENTIFY10', 'SNode_28', 'IDENTIFY9', 'TRY13', 'TRY14', 'TRY15', 'SNode_29', 'VAR20', 'SNode_31', 'SNode_10', 'SNode_33', 'GIVEN21', 'SNode_34', 'GOAL_56', 'APPLY32', 'GOAL_57', 'CHOOSE35', 'SNode_7', 'SNode_59', 'MAXIMIZE34', 'SNode_60', 'AXIS33', 'GOAL_61', 'WRITE31', 'GOAL_62', 'WRITE30', 'GOAL_63', 'RESOLVE37', 'SNode_64', 'NEED36', 'SNode_9', 'SNode_41', 'SNode_42', 'SNode_43', 'IDENTIFY39', 'GOAL_66', 'RESOLVE38', 'SNode_67', 'SNode_54', 'IDENTIFY41', 'GOAL_69', 'RESOLVE40', 'SNode_70', 'SNode_55', 'IDENTIFY43', 'GOAL_72', 'RESOLVE42', 'SNode_73', 'SNode_74', 'KINE29', 'SNode_4', 'SNode_75', 'VECTOR44', 'GOAL_79', 'EQUATION28', 'VECTOR27', 'RApp5', 'GOAL_80', 'RApp6', 'GOAL_81', 'TRY25', 'TRY24', 'GOAL_83', 'GOAL_84', 'CHOOSE47', 'SNode_86', 'SYSTEM46', 'SNode_156', 'NEWTONS45', 'GOAL_98', 'DEFINE23', 'SNode_37', 'IDENTIFY22', 'TRY26', 'SNode_38', 'SNode_40', 'SNode_44', 'SNode_46', 'SNode_65', 'NULL48', 'SNode_68', 'SNode_71', 'GOAL_87', 'FIND49', 'SNode_88', 'NORMAL50', 'NORMAL52', 'INCLINE51', 'SNode_91', 'SNode_12', 'SNode_13', 'STRAT_90', 'HORIZ53', 'BUGGY54', 'SNode_92', 'SNode_93', 'IDENTIFY55', 'SNode_94', 'WEIGHT56', 'SNode_95', 'WEIGHT57', 'SNode_97', 'GOAL_99', 'FIND58', 'SNode_100', 'IDENTIFY59', 'SNode_102', 'FORCE60', 'GOAL_103', 'APPLY61', 'GOAL_104', 'CHOOSE62', 'SNode_106', 'SNode_152', 'GOAL_107', 'WRITE63', 'GOAL_108', 'WRITE64', 'GOAL_109', 'GOAL_110', 'GOAL65', 'GOAL_111', 'GOAL66', 'NEED67', 'RApp7', 'RApp8', 'SNode_112', 'GOAL_113', 'GOAL68', 'GOAL_114', 'SNode_115', 'SNode_116', 'VECTOR69', 'SNode_117', 'SNode_118', 'VECTOR70', 'SNode_119', 'EQUAL71', 'SNode_120', 'GOAL_121', 'GOAL72', 'SNode_122', 'SNode_123', 'VECTOR73', 'SNode_124', 'NEWTONS74', 'SNode_125', 'SUM75', 'GOAL_126', 'GOAL_127', 'RApp9', 'RApp10', 'SNode_128', 'GOAL_129', 'GOAL_130', 'SNode_131', 'SNode_132', 'SNode_133', 'SNode_134', 'SNode_135', 'SNode_154', 'SNode_136', 'SNode_137', 'GOAL_142', 'GOAL_143', 'GOAL_146', 'RApp11', 'RApp12', 'RApp13', 'GOAL_147', 'GOAL_149', 'TRY76', 'GOAL_150', 'APPLY77', 'SNode_151', 'GRAV78', 'GOAL_153', 'SNode_155', 'SNode_14', 'SNode_18', 'SNode_19']
            generation_prompt = f"""We have found a causal system consisting of {len(node_names)} variables. The system can be described as follows:
            an Intelligent Tutoring System for Newtonian physics. ANDES’ student model uses a Bayesian network to do long-term knowledge assessment, plan recognition and prediction of students’ actions during problem solving. The network is updated in real time, using an approximate anytime algorithm based on stochastic sampling, as a student solves problems with ANDES. The information in the student model is used by ANDES’ Help system to tailor its support when the student reaches impasses in the problem solving process
            Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
            # descripton see Conati et al. Abstract
        case "sachs":
            node_names = ['Erk', 'Akt', 'PKA', 'Mek', 'Jnk', 'PKC', 'Raf', 'P38', 'PIP3', 'PIP2', 'Plcg']
            generation_prompt = f"""We have found a causal system consisting of {len(node_names)} variables. The system can be described as follows:
            A derivation of causal influences in cellular signaling networks. This derivation relied on the simultaneous measurement of multiple phosphorylated protein and phospholipid components in thousands of individual primary human immune system cells
            Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
            # descripton see sachs et al. 2005 abstract
        case "win95pts":
            node_names = ['AppOK', 'AppData', 'DataFile', 'EMFOK', 'DskLocal', 'PrtThread', 'GDIIN', 'PrtSpool', 'PrtDriver', 'GDIOUT', 'DrvSet', 'DrvOK', 'PrtDataOut', 'PrtSel', 'PrtPath', 'NetOK', 'NtwrkCnfg', 'PTROFFLINE', 'PrtCbl', 'LclOK', 'PrtPort', 'CblPrtHrdwrOK', 'DS_NTOK', 'PrtMpTPth', 'DS_LCLOK', 'NetPrint', 'PC2PRT', 'DSApplctn', 'PrtOn', 'PrtData', 'PrtPaper', 'PrtMem', 'PrtTimeOut', 'FllCrrptdBffr', 'TnrSpply', 'Problem1', 'AppDtGnTm', 'PrntPrcssTm', 'DeskPrntSpd', 'CmpltPgPrntd', 'PgOrnttnOK', 'PrntngArOK', 'NnPSGrphc', 'GrphcsRltdDrvrSttngs', 'EPSGrphc', 'PSGRAPHIC', 'Problem4', 'PrtPScript', 'TTOK', 'FntInstlltn', 'PrntrAccptsTrtyp', 'NnTTOK', 'ScrnFntNtPrntrFnt', 'TrTypFnts', 'Problem5', 'LclGrbld', 'NtGrbld', 'GrbldOtpt', 'HrglssDrtnAftrPrnt', 'REPEAT', 'AvlblVrtlMmry', 'PSERRMEM', 'TstpsTxt', 'GrbldPS', 'IncmpltPS', 'PrtFile', 'PrtIcon', 'Problem6', 'Problem3', 'NtSpd', 'PrtQueue', 'Problem2', 'PrtStatPaper', 'PrtStatToner', 'PrtStatMem', 'PrtStatOff']
            generation_prompt = f"""We have found a causal system consisting of {len(node_names)} variables. The system can be described as follows:
            An expert system for printer troubleshooting in Windows 95.
            Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
            # descripton see https://www.cs.huji.ac.il/w~galel/Repository/Datasets/win95pts/win95pts.htm
        case _:
            raise ValueError(f"{data} not found, check your spelling or implement this case.\nYou can alternaively run, run_llm_generic_prompt or run_llm_default")

        
    #run LLM
    pipeline = load_model_pipeline()
    out = text_generation(pipeline, generation_prompt, deterministic=deterministic)
    tags_string = text_reduction_variable_tags(pipeline, out, deterministic=deterministic)
    taglist_string = text_reduction_taglist(pipeline, out, deterministic=deterministic)
    #process LLM Output:
    tag_list = get_taglist_from_llm_output(taglist_string) #turn taglist to actual list
    tags = recover_tag_string_onevsall_using_taglist(tags_string=tags_string, tag_list=tag_list, node_names=node_names)
    
    return tags, node_names

def run_llm_generic_prompt(node_names : list, determinstic = True):
    """
    use this method to get tags using a LLM 
    :param prompt: prompt that is fed to the LLM, will independently then reduce it and put in the pipeline
    :returns tags: String with each line being the node name followed by colon and all tags seperated by by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :returns node_names: unchanged node_name input for consistency to other run_llm methods
    """

    #run LLM
    pipeline = load_model_pipeline()
    prompt = f"""We have found a causal system consisting of {len(node_names)} variables. Your job is now to assign those factors with multiple tags. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have **multiple** fitting tags. The variables are the following {node_names}"""
    print(prompt)
    out = text_generation(pipeline, prompt, determinstic)
    tags_string = text_reduction_variable_tags(pipeline, out, determinstic)
    taglist_string = text_reduction_taglist(pipeline, out, determinstic)
    #process LLM Output:
    tag_list = get_taglist_from_llm_output(taglist_string) #turn taglist to actual list
    tags = recover_tag_string_onevsall_using_taglist(tags_string=tags_string, tag_list=tag_list, node_names=node_names)
    
    return tags, node_names


def run_llm_default(prompt : str, node_names : list, determinstic = False):
    """
    use this method to get tags using a LLM 
    :param prompt: prompt that is fed to the LLM, will independently then reduce it and put in the pipeline
    """ #TODO returns einfügen

    #run LLM
    pipeline = load_model_pipeline()
    out = text_generation(pipeline, prompt, determinstic)
    tags_string = text_reduction_variable_tags(pipeline, out, determinstic)
    taglist_string = text_reduction_taglist(pipeline, out, determinstic)
    #process LLM Output:
    tag_list = get_taglist_from_llm_output(taglist_string) #turn taglist to actual list
    tags = recover_tag_string_onevsall_using_taglist(tags_string=tags_string, tag_list=tag_list, node_names=node_names)
    
    return tags, node_names