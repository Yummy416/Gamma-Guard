from datasets import load_dataset

ds1 = load_dataset("walledai/AdvBench", cache_dir='./cache')
ds2 = load_dataset("PKU-Alignment/BeaverTails", cache_dir='./cache')
ds3 = load_dataset("walledai/XSTest", cache_dir='./cache')
ds4 = load_dataset("toxigen/toxigen-data", "annotated", cache_dir='./cache')