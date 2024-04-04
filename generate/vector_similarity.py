import os
import json
import sys 
import numpy as np
from numpy.linalg import norm

sys.path.append('../')
sys.path.append('../eval/')

def get_indices_of_incontext_probs(prompt, k, incontext_list):
    
    question_list = []
    for file_name in incontext_list:
        with open(f"{file_name}/question.txt") as txt_file:
            question = txt_file.readlines()
        question_list.append("".join(question))
    
    correct_example_indices = []
    for i in range(1,k+1):
        incontext_example = prompt[prompt.find(f"PROBLEM {i}:"):prompt.find(f"PROBLEM {i+1}:")]
        indice_check = [1 if question in incontext_example else 0 for question in question_list]
        assert sum(indice_check) == 1, f"{sum(indice_check)} matches found for: {incontext_example}"
        correct_example_indices.append(np.argmax(indice_check))
        
    return correct_example_indices
    
def kl_divergence(p, q): # relative entropy from distirbution Q to P
    kl_divergence = 0
    assert len(p) == len(q), "distributions do not have the same # elements in sample space"
    for i in range(len(p)):
        if p[i] > 0:
            kl_divergence += p[i]*np.log(q[i]/p[i])
    return -1*kl_divergence

def main():

    dir_name = "round1_introductory"
    k = 2 # number of in-context examples selected per example
    incontext_file = "../data_split/10prob_train_introductory.json"
    with open(incontext_file) as json_file:
        incontext_list = json.load(json_file)
        
    train_file = "../data_split/train_introductory.json"
    with open(train_file) as json_file:
        train_list = json.load(json_file)

    problem_list = []
    for file_name in os.listdir(dir_name):
        if file_name.endswith(".json"):
            problem_list.append(file_name)
    
    problem_dict = {}
    for file_name in problem_list:
        with open(f"{dir_name}/{file_name}") as json_file:
            temp_dict = json.load(json_file)
        if 1.0 not in set(temp_dict["rewards"]):
            continue 
        
        index = int(file_name[file_name.find('-')+1:file_name.find('.json')])
        problem_path = train_list[index]
        #problem_num = problem_path[problem_path.rfind('/'):]
        problem_dict[file_name.replace(".json", "")] = temp_dict
        vector = [0]*len(incontext_list)
        for i in range(temp_dict["sample times"]):
            if temp_dict["rewards"][i] == 1.0:
                indices = get_indices_of_incontext_probs(temp_dict["prompts"][i], k, incontext_list)
                for idx in indices:
                    vector[idx] += 1
        normalized_vector = [float(item/sum(vector)) for item in vector]
        problem_dict[file_name.replace(".json", "")]["normalized_vector"] = vector
        problem_dict[file_name.replace(".json", "")]["problem_path"] = problem_path
        
    similarities = []
    target = problem_dict['sanjay-1361']
    target_vector = np.array(target["normalized_vector"])
    print("target path", target["problem_path"])
    for key in problem_dict:
        vector2 = np.array(problem_dict[key]["normalized_vector"])
        l1_distance = np.sum(np.abs(np.subtract(target_vector, vector2)))
        #kl_div = kl_divergence(target["normalized_vector"], problem_dict[key]["normalized_vector"])
        similarities.append((key, l1_distance, problem_dict[key]["problem_path"]))
    
    sorted_similarities = sorted(similarities, key=lambda tup: tup[1])
    print(sorted_similarities)
    5/0
    
            
        

if __name__ == "__main__":
    main()
