from allennlp.predictors.predictor import Predictor
from allennlp_models.structured_prediction.models import srl_bert
from allennlp_models import pretrained
#print(pretrained.get_pretrained_models())
predictor = pretrained.load_predictor('structured-prediction-srl-bert')
pred1=predictor.predict("Who placed the crab among the stars?")
pred2=predictor.predict("The crab bit Hercules on the foot, Hercules crushed it and then the goddess Hera, a sworn enemy of Hercules, placed the crab among the stars.")
print(pred1)
print(pred2)

#predictor = Predictor.from_path("bert-base-srl-2020.03.24.tar.gz")