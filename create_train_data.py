import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import sys
sys.path.append('./main')
import pandas as pd
from util import write_jsonl_file

datasets = pd.read_csv('datasets/beavertails/prompt_response_classified.csv').values.tolist()
datas = [{'question': question, 'response': response, 'attack_response': response} for question, response, label in datasets[:300]]
write_jsonl_file('datasets/attacks/common.jsonl', datas)