from utils import get_avg_saliency,get_gen_svg_saliency
import pandas as pd

df = pd.read_csv('/home/users/ybi3/SMLvsDL/SampleSplits_Sex/te_9194_rep_9.csv')
get_avg_saliency()
get_gen_svg_saliency(df, 'sex')

