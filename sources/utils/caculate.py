import os
from eval.rouge import Rouge
from eval.meteor import Meteor

model_name = "codebert"
datasets = [ "java"]
gold_name = "test.gold"
output_name = "test_beam10.output"

for dataset in datasets:
    gold_file = os.path.join(model_name, dataset, gold_name)
    output_file = os.path.join(model_name, dataset, output_name)
    with open(gold_file, encoding="utf-8") as gold, open(output_file,encoding="utf-8") as output:
        predictions = output.readlines()
        references = gold.readlines()
    assert len(predictions) == len(references)
    
    h_dict = dict() # idx : prediction
    r_dict = dict() # idx : ground truth
    for idx in range(len(predictions)):
        h_dict[idx] = [" ".join(predictions[idx].strip().split()[1:])]
        r_dict[idx] = [" ".join(references[idx].strip().split()[1:])]
    
    _, bleu, ind_bleu = corpus_bleu(h_dict, r_dict)
    
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(r_dict, h_dict)
    
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(r_dict, h_dict)
    print(bleu*100, rouge_l * 100, meteor * 100)
    break